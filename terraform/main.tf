provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required Google Cloud APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "run.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com",
    "aiplatform.googleapis.com",
  ])

  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Dedicated Least-Privilege Service Account for the ThermoFisher CoA Agent
resource "google_service_account" "agent_runtime_sa" {
  account_id   = "${var.service_name}-sa"
  display_name = "ThermoFisher Scientific CoA Agent Runtime Service Account"
  description  = "Least-privilege runtime identity for Cloud Run, Secret Manager, Vertex AI, and Cloud Trace"
}

# 3. Secret Manager Secret for Gemini API Key
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = var.gemini_secret_id

  replication {
    auto {}
  }

  labels = {
    app       = "thermofisher-coa-agent"
    managedby = "terraform"
    domain    = "quality-assurance-coa"
  }

  depends_on = [google_project_service.required_apis]
}

# 4. Grant Secret Manager Secret Accessor ONLY to the Agent Service Account
resource "google_secret_manager_secret_iam_member" "agent_secret_accessor" {
  secret_id = google_secret_manager_secret.gemini_api_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.agent_runtime_sa.email}"
}

# 5. Grant Cloud Trace, Cloud Logging, and Vertex AI User Roles
resource "google_project_iam_member" "agent_observability_roles" {
  for_each = toset([
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/aiplatform.user",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.agent_runtime_sa.email}"
}

# 6. Artifact Registry Docker Repository
resource "google_artifact_registry_repository" "agent_repo" {
  location      = var.region
  repository_id = "${var.service_name}-repo"
  description   = "Container repository for ThermoFisher CoA Chat Agent"
  format        = "DOCKER"

  depends_on = [google_project_service.required_apis]
}

# 7. Cloud Run v2 Service with Native Secret Manager Injection & Liveness Probes
resource "google_cloud_run_v2_service" "thermofisher_coa_service" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.agent_runtime_sa.email

    scaling {
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }

    containers {
      image = var.container_image

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "2"
          memory = "1Gi"
        }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "THERMOFISHER_GEMINI_SECRET_ID"
        value = var.gemini_secret_id
      }

      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }

      startup_probe {
        http_get {
          path = "/api/health"
          port = 8080
        }
        initial_delay_seconds = 3
        period_seconds        = 5
        failure_threshold     = 5
      }

      liveness_probe {
        http_get {
          path = "/api/health"
          port = 8080
        }
        period_seconds    = 15
        failure_threshold = 3
      }
    }
  }

  depends_on = [
    google_secret_manager_secret_iam_member.agent_secret_accessor,
    google_project_service.required_apis,
  ]
}
