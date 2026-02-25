Here is a clean, professional **Deployment Runbook** in Markdown format. You can copy this into a file named `DEPLOYMENT.md` in your project folder so you never have to troubleshoot these exact issues from scratch again.

---

```markdown
# Pytorch Web App Deployment Runbook (Azure VM)

This document summarizes the steps taken to successfully deploy the Pytorch application using Nginx and uWSGI on an Ubuntu-based Azure Virtual Machine.

---

## 1. Azure Networking (NSG)
To allow external traffic (IPv4 and IPv6) to reach the web server:
* **Port 80 (HTTP):** Must be open.
* **Source:** `Any` (Crucial for allowing IPv6 addresses from local ISP).
* **Protocol:** `TCP`.
* **Priority:** Ensure no 'Deny' rule has a lower number (higher priority).

## 2. Server Permissions
Nginx (running as `www-data`) needs "Execute" permissions on all parent directories to access the Unix socket.
```bash
sudo chmod +x /home/azureuser
sudo chmod +x /home/azureuser/pytorch-web-app-deploy-azure

```

## 3. uWSGI Configuration (`torchapp.ini`)

Key settings to ensure Nginx can communicate with the application:

```ini
socket = torchapp.sock
chmod-socket = 660
chown-socket = azureuser:www-data
vacuum = true
die-on-term = true

```

*Start command:* `sudo uwsgi --ini torchapp.ini`

## 4. Nginx Configuration

The configuration file should be located in `/etc/nginx/sites-available/` and linked to `sites-enabled/`.

**Important:** Disable the default Nginx landing page:

```bash
sudo rm /etc/nginx/sites-enabled/default

```

**Config Template:**

```nginx
server {
    listen 80;
    server_name 20.163.61.232; # Azure Public IP

    location / {
        include uwsgi_params;
        uwsgi_pass unix:/home/azureuser/pytorch-web-app-deploy-azure/torchapp.sock;
    }
}

```

## 5. Troubleshooting Commands

* **Check if Nginx is listening:** `sudo ss -tlnp | grep 80`
* **Test Port from Local:** `nc -zv -w 5 <Public_IP> 80`
* **Check Nginx Errors:** `sudo tail -f /var/log/nginx/error.log`
* **Check Socket Permissions:** `sudo -u www-data ls -l /path/to/torchapp.sock`

```

---

### What's next?
Now that your manual deployment is solid, you might eventually want to automate this using **Docker** or **GitHub Actions** so that every time you update your code, the server updates itself.

**Would you like me to explain how a Docker container could simplify this whole permission/socket mess in the future?**

```