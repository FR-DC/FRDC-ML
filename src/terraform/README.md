# Deployment Notes

The strategy for deploying Label Studio comes in two parts:
1. The Label Studio front-end 
2. The PostgreSQL back-end

## Initialization Strategy

The core strategy is to wrap **Label Studio** in a 
deployable Docker Image, publish it in a Container Registry
(like Google Container Registry). The image will have
necessary parameters to connect to the **PostgreSQL** database
which will be hosted separately.

1. Pre-define **Sensitive** information in `.env` and `secrets.tfvars`. 
2. Provision **PostgreSQL** through **Terraform**
3. Build **DockerFile** with connections retrieved from **Terraform Outputs**
4. Push **Docker Image** to **Container Registry**
5. Deploy **Docker Image**

## Choices of Hosts

We use [Supabase](https://supabase.com/) for our
PostgreSQL hosting solution as they have a generous free tier.
Furthermore, we don't foresee our annotations scaling beyond that.

For the Label Studio front-end, we use [Google Cloud Run](https://cloud.google.com/run)
as it is a serverless solution that scales to zero. This is important
as we don't want to pay for resources when the application is not in use.

