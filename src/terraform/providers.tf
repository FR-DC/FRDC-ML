terraform {
  required_providers {
    supabase = {
      source  = "supabase/supabase"
      version = "1.1.3"
    }
  }
}

provider "supabase" {
  access_token = var.supabase_access_token
}

resource "supabase_project" "production" {
  organization_id   = var.supabase_org_id
  name              = var.supabase_project_name
  database_password = var.db_password
  region            = var.supabase_project_region
  lifecycle {
    ignore_changes = [database_password]
  }
}