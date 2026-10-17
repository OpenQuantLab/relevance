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
pip install -e ".[serve]"
```

The code is a package under `src/relevance`, so it must be installed rather
than merely unpacked. `[serve]` pulls in gunicorn.

## 3. Test locally

```bash
gunicorn --bind 127.0.0.1:8050 --workers 2 --timeout 120 wsgi.demo:server
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
pip install -e ".[serve]"        # picks up dependency and entry-point changes
sudo systemctl restart relevance
```

### Viewing logs

```bash
sudo journalctl -u relevance -f
```

---

# Publishing Alpha Lab at datavik.no/projects/alpha-lab

Alpha Lab is a second app in the same folder. It reuses the venv and nginx you
already have, so this is five commands plus one config file.

Visitors get a **read-only dashboard**: they can browse every saved backtest,
chart and table, but the Run button stays locked until you type the password
into the sidebar. That keeps strangers off your CPU and your EODHD quota.

## 1. Settings file with the password

```bash
sudo cp /opt/relevance/deploy/alpha-lab.env.example /etc/alpha-lab.env
sudo nano /etc/alpha-lab.env        # set ALPHA_LAB_PASSWORD and EODHD_API_TOKEN
sudo chown root:www-data /etc/alpha-lab.env
sudo chmod 640 /etc/alpha-lab.env
```

Pick a long random password. To generate one: `openssl rand -base64 24`.

## 2. Let the app write its results folder

```bash
sudo mkdir -p /opt/relevance/lab
sudo chown -R www-data:www-data /opt/relevance/lab
```

Saved runs live in `/opt/relevance/lab/runs/` and cached prices in
`/opt/relevance/lab/cache/`. Back up `runs/` if the results matter to you.

## 3. Start the service

```bash
sudo cp /opt/relevance/deploy/alpha-lab.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now alpha-lab
sudo systemctl status alpha-lab     # should say "active (running)"
```

## 4. Tell nginx about it

The location block is already in `deploy/nginx-relevance.conf`. If you copied
that file earlier, re-copy it; if you merged it into your own server block, add
the `/projects/alpha-lab/` block the same way you added the relevance one.

```bash
sudo cp /opt/relevance/deploy/nginx-relevance.conf /etc/nginx/sites-available/relevance
sudo nginx -t
sudo systemctl reload nginx
```

If you use HTTPS, the block must also exist in your port-443 server block.

## 5. Share the link

`https://datavik.no/projects/alpha-lab/`

Anyone can open it and explore your runs. When you want to run a new backtest,
type the password into the sidebar — it unlocks for your browser only, and
stays unlocked until you close the tab.

---

### Seeding it with results

An empty dashboard is a dull first impression. Run a few backtests locally and
copy them up before you share the link:

```bash
rsync -av lab/runs/ your-user@datavik.no:/opt/relevance/lab/runs/
ssh your-user@datavik.no "sudo chown -R www-data:www-data /opt/relevance/lab"
```

### Updating Alpha Lab

```bash
cd /opt/relevance && git pull
sudo systemctl restart alpha-lab
```

### Logs and troubleshooting

```bash
sudo journalctl -u alpha-lab -f
```

| Symptom | Fix |
| --- | --- |
| Blank page, 404s on `.js` files | `ALPHA_LAB_URL_PREFIX` doesn't match the nginx location. Both must be `/projects/alpha-lab/`, with both slashes. |
| "Permission denied" in the logs | `/opt/relevance/lab` isn't owned by `www-data` — redo step 2. |
| Password never works | The service didn't read `/etc/alpha-lab.env`. Check `sudo systemctl show alpha-lab -p EnvironmentFiles`. |
| Backtest never finishes | Check the EODHD token is valid; the first run of a universe downloads prices. |

### A note on the API key

`eodhd_data.py` has a fallback token committed in the source. Setting
`EODHD_API_TOKEN` in `/etc/alpha-lab.env` overrides it, and you should rotate
the committed one at some point since it is public in the repo.
