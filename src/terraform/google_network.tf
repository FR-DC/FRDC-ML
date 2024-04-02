resource "google_compute_firewall" "label-studio-port" {
  name    = "label-studio-port"
  network = google_compute_network.label-studio-vpc.name

  allow {
    protocol = "tcp"
    ports    = ["22", "80", "443", "8080"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags = ["label-studio"]
}

resource "google_compute_network" "label-studio-vpc" {
  name                    = "label-studio-vpc"
  auto_create_subnetworks = true
}