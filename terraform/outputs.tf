output "cloud_run_service_url" {
  description = "Deployed URL of the ThermoFisher CoA Chat Agent Cloud Run v2 service"
  value       = google_cloud_run_v2_service.thermofisher_coa_service.uri
}

output "service_account_email" {
  description = "Least-privilege runtime service account email"
  value       = google_service_account.agent_runtime_sa.email
}

output "secret_manager_secret_id" {
  description = "Secret Manager resource ID for the Gemini API key"
  value       = google_secret_manager_secret.gemini_api_key.id
}

output "artifact_registry_repository" {
  description = "Artifact Registry Docker repository name"
  value       = google_artifact_registry_repository.agent_repo.name
}
