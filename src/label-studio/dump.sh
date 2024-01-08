echo "Creating backups directory..."
docker exec label-studio-db-1 sh -c "if [ ! -d \"/var/lib/postgresql/backups/\" ]; then mkdir -p \"/var/lib/postgresql/backups/\"; fi"

echo "Checking if label-studio-db-1 is running..."
docker exec label-studio-db-1 sh -c "pg_isready -U postgres"

if [ $? -ne 0 ]; then
    echo "label-studio-db-1 is not running. Exiting..."
    exit 1
fi

echo "Dumping database... to /var/lib/postgresql/backups/"
docker exec label-studio-db-1 sh -c "pg_dump -Fc -U postgres -d postgres -f \"/var/lib/postgresql/backups/$(date +'%d-%m-%Y_%HH%MM%SS').backup\""

echo "Dumping database in SQL format... to /var/lib/postgresql/backups/"
docker exec label-studio-db-1 sh -c "pg_dump -U postgres -d postgres -f \"/var/lib/postgresql/backups/$(date +'%d-%m-%Y_%HH%MM%SS').sql\""
