variable "google_project" {
  description = "The Google Cloud project to use"
  type        = string
  default     = "frmodel"
}

variable "google_region" {
  description = "The Google Cloud region to use"
  type        = string
  default     = "asia-southeast1"
}

variable "google_zone" {
  description = "The Google Cloud zone to use"
  type        = string
  default     = "asia-southeast1-b"
}

variable "google_ce_machine_type" {
  description = "The Google Cloud Compute Engine machine type to use"
  type        = string
  default     = "n1-standard-1"
}
variable "label_studio_username" {
  description = "The root Label Studio username (Must be an email)"
  type        = string
  sensitive   = true
}

variable "label_studio_password" {
  description = "The root Label Studio password"
  type        = string
  sensitive   = true
}