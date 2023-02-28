echo "Testing Docker installation"
groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world