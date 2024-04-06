resource "google_compute_firewall" "label-studio-port" {
  name    = var.google_firewall_name
  network = google_compute_network.label-studio-vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["label-studio"]
  depends_on = [
    google_compute_network.label-studio-vpc
  ]
}

resource "google_compute_network" "label-studio-vpc" {
  name                    = var.google_network_name
  auto_create_subnetworks = true
}

resource "google_compute_address" "label-studio-ip" {
  name = var.google_ip_name
}