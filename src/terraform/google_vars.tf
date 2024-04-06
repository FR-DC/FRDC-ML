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

variable "google_compute_machine_type" {
  description = "The Google Cloud Compute Engine machine type to use"
  type        = string
  default     = "n1-standard-1"
}

variable "google_compute_name" {
  description = "The Google Cloud Compute Engine instance name"
  type        = string
  default     = "label-studio"
}

variable "google_vpc_name" {
  description = "The Google Cloud (VPC) network name"
  type        = string
  default     = "label-studio-vpc"
}

variable "google_firewall_name" {
  description = "The Google Cloud firewall name"
  type        = string
  default     = "label-studio-port"
}

variable "google_address_name" {
  description = "The Google Cloud static IP name"
  type        = string
  default     = "label-studio-ip"
}

variable "ls_username" {
  description = "The root Label Studio username (Must be an email)"
  type        = string
  sensitive   = true
}

variable "ls_password" {
  description = "The root Label Studio password"
  type        = string
  sensitive   = true
}