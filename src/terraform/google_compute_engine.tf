# This code is compatible with Terraform 4.25.0 and versions that are backward compatible to 4.25.0.
# For information about validating this Terraform code, see https://developer.hashicorp.com/terraform/tutorials/gcp-get-started/google-cloud-platform-build#format-and-validate-the-configuration

resource "google_compute_instance" "label-studio" {
  boot_disk {
    auto_delete = true
    device_name = "label-studio"

    initialize_params {
      image = "projects/cos-cloud/global/images/cos-stable-109-17800-147-38"
      size  = 10
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  labels = {
    container-vm = "cos-stable-109-17800-147-38"
    goog-ec-src  = "vm_add-tf"
  }

  machine_type = var.google_ce_machine_type

  metadata = {
    enable-oslogin            = "true"
    gce-container-declaration = <<-EOF
spec:
    containers:
    - name: label-studio
      image: ghcr.io/fr-dc/frdc-ml:terraform
      env:
      - name: POSTGRE_PASSWORD
        value: ${var.db_password}
      - name: POSTGRE_USER
        value: postgres.${supabase_project.production.id}
      - name: POSTGRE_HOST
        value: aws-0-${supabase_project.production.region}.pooler.supabase.com
      - name: LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK
        value: true
      - name: LABEL_STUDIO_USERNAME
        value: ${var.label_studio_username}
      - name: LABEL_STUDIO_PASSWORD
        value: ${var.label_studio_password}
      stdin: false
      tty: false
    restartPolicy: Always
EOF
  }

  name = "label-studio"

  network_interface {
    access_config {
      network_tier = "STANDARD"
    }

    queue_count = 0
    stack_type  = "IPV4_ONLY"
    subnetwork  = google_compute_network.label-studio-vpc.name
  }

  scheduling {
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
    preemptible         = true
    provisioning_model  = "SPOT"
  }

  service_account {
    email = "673270019389-compute@developer.gserviceaccount.com"
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_only",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/service.management.readonly",
      "https://www.googleapis.com/auth/servicecontrol",
      "https://www.googleapis.com/auth/trace.append"
    ]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  zone = var.google_zone

  tags = ["label-studio"]
}

