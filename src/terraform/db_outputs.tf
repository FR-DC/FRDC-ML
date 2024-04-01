output "pg_user" {
  value = "postgres.${supabase_project.production.id}"
}

output "pg_host" {
  value = "aws-0-${supabase_project.production.region}.pooler.supabase.com"
}