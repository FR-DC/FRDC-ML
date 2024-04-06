variable "supabase_access_token" {
  description = "The access token for Supabase"
  type        = string
  sensitive   = true
}

variable "supabase_org_id" {
  description = "The organization ID for Supabase"
  type        = string
  sensitive   = true
}

variable "supabase_project_id" {
  description = "The project ID for Supabase"
  type        = string
  sensitive   = true
}

variable "supabase_project_name" {
  description = "The project name for Supabase"
  type        = string
  default     = "frdc"
}

variable "supabase_project_region" {
  description = "The project region for Supabase"
  type        = string
  default     = "ap-southeast-1"
}

variable "db_password" {
  description = "The password for the database"
  type        = string
  sensitive   = true
}
