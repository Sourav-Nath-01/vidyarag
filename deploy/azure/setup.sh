#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy/azure/setup.sh — one-time Azure provisioning for the VidyaRAG API
#
# Run it in Azure Cloud Shell (Bash) at https://shell.azure.com, AFTER the
# first GitHub Actions run on main has pushed the image to GHCR:
#
#   curl -fsSL https://raw.githubusercontent.com/Sourav-Nath-01/vidyarag/main/deploy/azure/setup.sh | bash
#
# Override any default with an env var, e.g.  LOCATION=southeastasia bash setup.sh
#
# Everything lives in ONE resource group, so cleanup is a single command:
#   az group delete --name rg-vidyarag --yes
#
# What it creates
#   • Container Apps environment (+ Log Analytics workspace for logs)
#   • Container App: 2 vCPU / 4 GiB, scale 0→1, HTTPS ingress, health probes
#   • User-assigned managed identity + GitHub OIDC federated credential, so the
#     GitHub Actions pipeline can deploy WITHOUT any stored password or key
#
# Safe to re-run (idempotent).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

LOCATION="${LOCATION:-centralindia}"
RG="${RG:-rg-vidyarag}"
ENV_NAME="${ENV_NAME:-cae-vidyarag}"
APP="${APP:-vidyarag-api}"
IDENTITY="${IDENTITY:-id-vidyarag-github}"
GITHUB_REPO="${GITHUB_REPO:-Sourav-Nath-01/vidyarag}"
IMAGE="${IMAGE:-ghcr.io/sourav-nath-01/vidyarag-api:latest}"

say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

SUB_ID="$(az account show --query id -o tsv)"
TENANT_ID="$(az account show --query tenantId -o tsv)"
say "Subscription: $(az account show --query name -o tsv)"

# Azure for Students subscriptions only allow some regions — check up front.
ALLOWED="$(az policy assignment list \
  --query "[?parameters.listOfAllowedLocations].parameters.listOfAllowedLocations.value[]" \
  -o tsv 2>/dev/null | sort -u | tr '\n' ' ' || true)"
if [[ -n "${ALLOWED// /}" ]] && ! grep -qw "$LOCATION" <<<"$ALLOWED"; then
  echo "Region '$LOCATION' is blocked by a policy on this subscription."
  echo "Allowed regions: $ALLOWED"
  echo "Re-run with one of them, e.g.:  LOCATION=<region> bash setup.sh"
  exit 1
fi

az config set extension.use_dynamic_install=yes_without_prompt --only-show-errors >/dev/null
az extension add --name containerapp --upgrade --only-show-errors >/dev/null

say "Registering resource providers (first run only — can take a few minutes)"
for ns in Microsoft.App Microsoft.OperationalInsights Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --wait --only-show-errors
done

say "Resource group: $RG ($LOCATION)"
az group create --name "$RG" --location "$LOCATION" --output none

say "Container Apps environment: $ENV_NAME"
if ! az containerapp env show --name "$ENV_NAME" --resource-group "$RG" --output none 2>/dev/null; then
  az containerapp env create --name "$ENV_NAME" --resource-group "$RG" \
    --location "$LOCATION" --output none
fi
ENV_ID="$(az containerapp env show --name "$ENV_NAME" --resource-group "$RG" --query id -o tsv)"

say "Container App: $APP  ←  $IMAGE"
SPEC="$(mktemp --suffix .yaml)"
cat > "$SPEC" <<YAML
location: ${LOCATION}
properties:
  managedEnvironmentId: ${ENV_ID}
  configuration:
    activeRevisionsMode: Single
    ingress:
      external: true
      targetPort: 7860
      transport: auto
      allowInsecure: false
  template:
    containers:
      - name: api
        image: ${IMAGE}
        resources:
          cpu: 2.0
          memory: 4Gi
        probes:
          - type: Startup        # loading BGE-large + the C3 index takes ~30-60 s
            httpGet:
              path: /health
              port: 7860
            initialDelaySeconds: 20
            periodSeconds: 15
            timeoutSeconds: 5
            failureThreshold: 10
          - type: Readiness      # only route traffic once warm-up has succeeded
            httpGet:
              path: /ready
              port: 7860
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          - type: Liveness       # restart the container if it stops responding
            httpGet:
              path: /health
              port: 7860
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
    scale:
      minReplicas: 0             # scale to zero when idle → pay only while in use
      maxReplicas: 1             # hard cap on spend
      rules:
        - name: http-traffic
          http:
            metadata:
              concurrentRequests: "10"
YAML
if az containerapp show --name "$APP" --resource-group "$RG" --output none 2>/dev/null; then
  az containerapp update --name "$APP" --resource-group "$RG" --yaml "$SPEC" --output none
else
  az containerapp create --name "$APP" --resource-group "$RG" --yaml "$SPEC" --output none
fi
rm -f "$SPEC"
FQDN="$(az containerapp show --name "$APP" --resource-group "$RG" \
          --query properties.configuration.ingress.fqdn -o tsv)"

say "Keyless GitHub → Azure auth (managed identity + OIDC federated credential)"
az identity create --name "$IDENTITY" --resource-group "$RG" --location "$LOCATION" --output none
CLIENT_ID="$(az identity show --name "$IDENTITY" --resource-group "$RG" --query clientId -o tsv)"
PRINCIPAL_ID="$(az identity show --name "$IDENTITY" --resource-group "$RG" --query principalId -o tsv)"

# Only workflow jobs running in this repo's "production" environment can use it.
if ! az identity federated-credential show --name github-production \
      --identity-name "$IDENTITY" --resource-group "$RG" --output none 2>/dev/null; then
  az identity federated-credential create --name github-production \
    --identity-name "$IDENTITY" --resource-group "$RG" \
    --issuer "https://token.actions.githubusercontent.com" \
    --subject "repo:${GITHUB_REPO}:environment:production" \
    --audiences "api://AzureADTokenExchange" --output none
fi

# Least privilege: Contributor on this resource group only, not the subscription.
RG_ID="$(az group show --name "$RG" --query id -o tsv)"
for attempt in 1 2 3 4 5 6; do   # a new identity can take a minute to replicate
  if az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
       --assignee-principal-type ServicePrincipal --role Contributor \
       --scope "$RG_ID" --output none 2>/dev/null; then
    break
  fi
  if [[ $attempt -eq 6 ]]; then echo "Could not create the role assignment — re-run the script."; exit 1; fi
  sleep 15
done

cat <<EOF

────────────────────────────────────────────────────────────────────────────
✅ Azure is ready.

   API:       https://${FQDN}
   API docs:  https://${FQDN}/docs
   (first request after idle takes ~1-2 min while the container cold-starts)

Turn on automatic deploys from GitHub — run these where 'gh' is logged in:

   gh secret set AZURE_CLIENT_ID       --body "${CLIENT_ID}" -R ${GITHUB_REPO}
   gh secret set AZURE_TENANT_ID       --body "${TENANT_ID}" -R ${GITHUB_REPO}
   gh secret set AZURE_SUBSCRIPTION_ID --body "${SUB_ID}" -R ${GITHUB_REPO}
   gh variable set AZURE_RESOURCE_GROUP    --body "${RG}"  -R ${GITHUB_REPO}
   gh variable set AZURE_CONTAINERAPP_NAME --body "${APP}" -R ${GITHUB_REPO}
   gh variable set AZURE_DEPLOY_ENABLED    --body true     -R ${GITHUB_REPO}

(These are identifiers, not passwords — the login is keyless OIDC.)
────────────────────────────────────────────────────────────────────────────
EOF
