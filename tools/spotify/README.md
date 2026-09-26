# 🎵 Spotify Playlist Downloader & Player Guide

Welcome! This folder contains everything you need to batch-download Spotify playlists and play them offline with a premium player.

---

## ⚡ Part 1: How to Download Playlists (Automated)

Since music download websites only support downloading songs "one by one", we use a browser script that automatically clicks and downloads all songs for you sequentially.

### Step-by-Step Instructions:

1. **Open the Website:**
   Open Firefox and go to:
   👉 [https://spotifymp3.com/spotify-playlist-downloader](https://spotifymp3.com/spotify-playlist-downloader)

2. **Load your Playlist:**
   Paste your Spotify playlist link in the input box and click **Download**. Wait until the complete list of songs is displayed on the page.

3. **Open the Web Console:**
   * On Windows: Press **`Ctrl + Shift + K`**
   * On Mac: Press **`Cmd + Option + K`**
   *(Or right-click anywhere on the page, click **Inspect**, and switch to the **Console** tab).*

4. **Firefox Protection Bypass (First-time only):**
   If Firefox warns you about pasting, type **`allow pasting`** into the console and press **Enter**.

5. **Run the Script:**
   Open the file [spotify_download_script.txt](spotify_download_script.txt) in this folder, copy all of the code, paste it into the console, and press **Enter**.

6. **Watch it Work:**
   * A **green progress banner** will appear on the web page showing you which song is downloading (e.g., `Song 5 of 20`).
   * **Important:** Keep the browser tab open while it runs!
   * The browser will automatically save all files to your **Downloads** folder.
   * *If Firefox asks for permission to download multiple files, click **Allow**.*

---

## 🎨 Part 2: How to Play your Music (RetroWave Player)

We created a custom offline audio player in this folder so you can listen to your downloaded songs with a real-time audio visualizer.

### How to use it:

1. Double-click the file [index.html](index.html) in this folder. It will open in your web browser.
2. Select your downloaded `.mp3` files from your computer's **Downloads** folder.
3. **Drag and drop** those files directly onto the RetroWave player screen.
4. Select a song from the **Queue** on the right, click **Play**, and enjoy your music!

---

## 💡 Troubleshooting & Tips

* **Why is there a 10-second delay between downloads?**
  We set a 10-second wait timer between each download in the script. This is intentional. If the script downloads too fast, the website's security (Cloudflare) will think you are a bot and block your IP address.
* **Can I download other playlists?**
  Yes! Just paste a new playlist link on the website, click Download, open the console, and paste the script again.
