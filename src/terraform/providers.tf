terraform {
  required_providers {
    supabase = {
      source  = "supabase/supabase"
      version = "1.1.3"
    }
    google = {
      source  = "hashicorp/google"
      version = "5.23.0"
    }
  }
}

provider "supabase" {
  access_token = var.supabase_access_token
}

provider "google" {
  project = var.google_project
  region  = var.google_region
  zone    = var.google_zone
}
