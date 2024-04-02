output "db_user" {
  value = "postgres.${supabase_project.production.id}"
}

output "db_host" {
  value = "aws-0-${supabase_project.production.region}.pooler.supabase.com"
}

output "ls_internal_ip" {
  value = google_compute_instance.label-studio.network_interface[0].network_ip
}
output "ls_external_ip" {
  value = google_compute_instance.label-studio.network_interface[0].access_config[0].nat_ip
}
