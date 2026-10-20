variable "project_id" {
  description = "Google Cloud Project ID for deploying the ThermoFisher CoA Chat Agent"
  type        = string
}

variable "region" {
  description = "Primary GCP region for Cloud Run and Artifact Registry"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run v2 service name"
  type        = string
  default     = "thermofisher-coa-agent"
}

variable "container_image" {
  description = "Container image URI in Artifact Registry"
  type        = string
  default     = "us-central1-docker.pkg.dev/example-project/thermofisher-coa-repo/thermofisher-coa-agent:latest"
}

variable "gemini_secret_id" {
  description = "GCP Secret Manager secret ID holding the Gemini API key"
  type        = string
  default     = "thermofisher-gemini-api-key"
}

variable "min_instance_count" {
  description = "Minimum Cloud Run instances"
  type        = number
  default     = 0
}

variable "max_instance_count" {
  description = "Maximum Cloud Run instances"
  type        = number
  default     = 10
}
