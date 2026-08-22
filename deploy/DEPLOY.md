# Deploying to datavik.no/projects/relevance

## 1. Copy the code to the server

```bash
ssh your-user@datavik.no
sudo mkdir -p /opt/relevance
# scp, rsync, or git clone your repo into /opt/relevance
```

## 2. Set up Python environment

```bash
cd /opt/relevance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Test locally

```bash
gunicorn --bind 127.0.0.1:8050 --workers 2 --timeout 120 app:server
# Visit http://server-ip:8050/projects/relevance/ to confirm it works
```

## 4. Install the systemd service

```bash
sudo cp deploy/relevance.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable relevance
sudo systemctl start relevance
sudo systemctl status relevance   # confirm it's running
```

## 5. Install nginx

```bash
sudo apt install nginx          # Debian/Ubuntu
# sudo dnf install nginx        # Fedora/RHEL
```

## 6. Configure nginx

```bash
sudo cp deploy/nginx-relevance.conf /etc/nginx/sites-available/relevance
sudo ln -s /etc/nginx/sites-available/relevance /etc/nginx/sites-enabled/
sudo nginx -t                   # test config
sudo systemctl reload nginx
```

If you already have a server block for `datavik.no`, merge the
`location /projects/relevance/` block into it instead.

## 7. (Optional) HTTPS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d datavik.no -d www.datavik.no
```

## 8. Verify

Open `https://datavik.no/projects/relevance/` in a browser.

---

### Updating the app

```bash
cd /opt/relevance
git pull                         # or re-upload files
source venv/bin/activate
pip install -r requirements.txt  # if deps changed
sudo systemctl restart relevance
```

### Viewing logs

```bash
sudo journalctl -u relevance -f
```
