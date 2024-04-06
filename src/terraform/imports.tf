import {
  id = "projects/${var.google_project}/zones/${var.google_zone}/instances/${var.google_ce_name}"
  to = google_compute_instance.label-studio
}
import {
  id = "projects/${var.google_project}/global/firewalls/${var.google_firewall_name}"
  to = google_compute_firewall.label-studio-port
}
import {
  id = "projects/${var.google_project}/global/networks/${var.google_network_name}"
  to = google_compute_network.label-studio-vpc
}
import {
  id = "projects/${var.google_project}/regions/${var.google_region}/addresses/${var.google_ip_name}"
  to = google_compute_address.label-studio-ip
}
import {
  to = supabase_project.production
  id = var.supabase_project_id
}
