(function() {
    "use strict";

    const safeText = (window.ZevBuildDom && window.ZevBuildDom.setText) ? window.ZevBuildDom.setText : function(el, txt) { if (el) el.textContent = txt; };
    let accessAllowed = true;
    const authGate = (window.ZevBuildAuthGate && window.ZevBuildAuthGate.create) ? window.ZevBuildAuthGate.create({
      previewKey: "zevbuild_upi_preview_started_at",
      trialPill: document.getElementById("trial-pill"),
      trialTime: document.getElementById("trial-time"),
      accessLock: document.getElementById("access-lock"),
      onUnlock(user) {
        accessAllowed = true;
        const pill = document.getElementById("trial-pill");
        if (pill) pill.style.display = user ? "none" : "flex";
      },
      onLock() {
        accessAllowed = false;
        if (typeof window.stopCameraScan === "function") window.stopCameraScan();
      },
      onRequireLogin() {
        showToast("Login required to continue", "error");
      },
      onVerifiedUser(user) {
        showToast(`Welcome back, ${user.email || "ZevBuild user"}`, "success");
      }
    }) : {
      requireAccess() { return true; }
    };

    function requireAccess() {
      if (accessAllowed) return true;
      return authGate.requireAccess();
    }
    // STATE
    let currentTab = 'manual';
    let generatedLink = '';
    let paymentDone = false;
    let scannedUPI = '';
    let cameraStream = null;
    let scanInterval = null;
    const MAX_QR_FILE_BYTES = 4 * 1024 * 1024;

    //  RECENT UPI (localStorage) 
    function getRecent() {
      try { return JSON.parse(localStorage.getItem('upi_recent_v2') || '[]'); } catch { return []; }
    }
    function saveRecent(upiId, name) {
      let arr = getRecent().filter(x => x.id !== upiId);
      arr.unshift({ id: upiId, name: name || '' });
      if (arr.length > 6) arr = arr.slice(0, 6);
      localStorage.setItem('upi_recent_v2', JSON.stringify(arr));
    }
    window.clearRecent = function() {
      localStorage.removeItem('upi_recent_v2');
      renderRecent();
      showToast(' Recent UPI IDs cleared', 'success');
    };
    function removeRecent(upiId) {
      const arr = getRecent().filter(x => x.id !== upiId);
      localStorage.setItem('upi_recent_v2', JSON.stringify(arr));
      renderRecent();
    }
    function renderRecent() {
      const arr = getRecent();
      const sec = document.getElementById('recent-section');
      const chipsContainer = document.getElementById('recent-chips');
      if (!arr.length) { sec.style.display = 'none'; return; }
      sec.style.display = 'block';
      chipsContainer.innerHTML = '';
      arr.forEach(r => {
        const chip = document.createElement('div');
        chip.className = 'recent-chip';
        chip.addEventListener('click', () => useRecent(r.id, r.name));
        const iconSpan = document.createElement('span');
        iconSpan.textContent = ' ';
        chip.appendChild(iconSpan);
        chip.appendChild(document.createTextNode(r.id));
        const delSpan = document.createElement('span');
        delSpan.className = 'recent-chip-del';
        delSpan.textContent = '';
        delSpan.addEventListener('click', (e) => { e.stopPropagation(); removeRecent(r.id); });
        chip.appendChild(delSpan);
        chipsContainer.appendChild(chip);
      });
    }
    function showRecent() { renderRecent(); }
    window.useRecent = function(id, name) {
      document.getElementById('upi-id').value = id;
      if (name) document.getElementById('payee-name').value = name;
      validateUPI(document.getElementById('upi-id'));
    };

    //  TAB SWITCH 
    window.switchTab = function(tab) {
      currentTab = tab;
      stopCameraScan();
      document.getElementById('panel-manual').style.display = tab === 'manual' ? 'block' : 'none';
      document.getElementById('panel-qr').style.display = tab === 'qr' ? 'block' : 'none';
      document.getElementById('tab-manual').classList.toggle('active', tab === 'manual');
      document.getElementById('tab-qr').classList.toggle('active', tab === 'qr');
    };

    //  VALIDATION 
    function validateUPI(input) {
      const val = input.value.trim();
      const pattern = /^[\w.\-+]+@[\w]+$/;
      if (!val) { input.classList.remove('error', 'valid'); return false; }
      if (pattern.test(val)) { input.classList.remove('error'); input.classList.add('valid'); return true; }
      else { input.classList.remove('valid'); input.classList.add('error'); return false; }
    }
    window.validateUPI = validateUPI;
    window.validateAmount = function(input) {
      const val = parseFloat(input.value);
      if (!isNaN(val) && val > 0) input.classList.remove('error');
      else input.classList.add('error');
    };
    window.setAmount = function(val, panel = 'manual') {
      document.getElementById(panel === 'qr' ? 'qr-amount' : 'amount').value = val;
    };

    //  QR HELPERS 
    function parseUPIData(data) {
      try {
        if (data.startsWith('upi://')) {
          const url = new URL(data);
          const params = {};
          url.searchParams.forEach((v, k) => params[k] = v);
          return params;
        }
        if (/^[\w.\-+]+@[\w]+$/.test(data.trim())) return { pa: data.trim() };
        return null;
      } catch { return null; }
    }
    function showQRResult(message, isError = false) {
      const el = document.getElementById('qr-scan-result');
      el.className = 'qr-scan-result' + (isError ? ' error' : '');
      el.style.display = 'block';
      el.textContent = message; // safe
    }
    function applyScannedUPI(upiData) {
      if (!upiData || !upiData.pa) return;
      scannedUPI = upiData.pa;
      document.getElementById('qr-name').value = upiData.pn || '';
      if (upiData.am) document.getElementById('qr-amount').value = upiData.am;
      document.getElementById('upi-id').value = upiData.pa;
      if (upiData.pn) document.getElementById('payee-name').value = upiData.pn;
      if (upiData.am) document.getElementById('amount').value = upiData.am;
      validateUPI(document.getElementById('upi-id'));
      showToast(' UPI details filled', 'success');
    }

    //  CAMERA SCANNER 
    window.startCameraScan = async function() {
      if (!requireAccess()) return;
      if (typeof jsQR !== 'function') {
        showToast('QR scanner is still loading. Try again in a moment.', 'error');
        return;
      }
      try {
        stopCameraScan();
        if (!window.isSecureContext) {
          showToast(' Camera requires HTTPS', 'error'); return;
        }
        const video = document.getElementById('camera-preview');
        cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = cameraStream;
        video.style.display = 'block';
        document.getElementById('stop-camera-btn').style.display = 'block';
        document.getElementById('start-camera-btn').style.display = 'none';
        await video.play();
        const canvas = document.getElementById('camera-canvas');
        const ctx = canvas.getContext('2d');
        scanInterval = setInterval(() => {
          if (video.readyState === video.HAVE_ENOUGH_DATA) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const code = jsQR(imageData.data, imageData.width, imageData.height);
            if (code) {
              const upiData = parseUPIData(code.data);
              if (upiData && upiData.pa) {
                showQRResult(` Scanned: ${upiData.pa}`);
                applyScannedUPI(upiData);
                vibrate([50,30,50]);
                stopCameraScan();
              }
            }
          }
        }, 200);
      } catch (err) {
        showToast(' Camera access denied or not available', 'error');
        console.error(err);
      }
    };
    window.stopCameraScan = function() {
      if (scanInterval) { clearInterval(scanInterval); scanInterval = null; }
      if (cameraStream) { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }
      const video = document.getElementById('camera-preview');
      video.style.display = 'none';
      video.srcObject = null;
      document.getElementById('stop-camera-btn').style.display = 'none';
      document.getElementById('start-camera-btn').style.display = 'block';
    };

    //  FILE QR SCAN 
    window.handleQRDrop = function(e) {
      e.preventDefault();
      if (!requireAccess()) return;
      e.currentTarget.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('image/')) processQRFile(file);
    };
    window.handleQRUpload = function(e) {
      if (!requireAccess()) return;
      const file = e.target.files[0];
      if (file) processQRFile(file);
    };
    function processQRFile(file) {
      if (!requireAccess()) return;
      if (typeof jsQR !== 'function') {
        showToast('QR scanner is still loading. Try again in a moment.', 'error');
        return;
      }
      if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
      }
      if (file.size > MAX_QR_FILE_BYTES) {
        showToast('QR image must be under 4 MB', 'error');
        return;
      }
      const reader = new FileReader();
      reader.onload = function(e) {
        const src = e.target.result;
        document.getElementById('qr-preview-img').src = src;
        document.getElementById('qr-preview-container').style.display = 'block';
        const img = new Image();
        img.onload = function() {
          const canvas = document.getElementById('qr-scan-canvas');
          canvas.width = img.width; canvas.height = img.height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0);
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const code = jsQR(imageData.data, imageData.width, imageData.height);
          if (code) {
            const upiData = parseUPIData(code.data);
            if (upiData && upiData.pa) {
              showQRResult(` QR scanned: ${upiData.pa}`);
              applyScannedUPI(upiData);
              vibrate([50,30,50]);
            } else {
              showQRResult(` QR found but not a UPI code. Data: ${code.data.substring(0, 60)}`, true);
            }
          } else {
            showQRResult(' Could not read QR code. Try a clearer image.', true);
          }
        };
        img.src = src;
      };
      reader.readAsDataURL(file);
    }

    //  GENERATE FLOW 
    window.handleGenerate = function() {
      if (!requireAccess()) return;
      const upiId = currentTab === 'manual' ? document.getElementById('upi-id').value.trim() : (scannedUPI || '');
      const name = currentTab === 'manual' ? document.getElementById('payee-name').value.trim() : document.getElementById('qr-name').value.trim();
      const amount = currentTab === 'manual' ? document.getElementById('amount').value.trim() : document.getElementById('qr-amount').value.trim();
      const note = currentTab === 'manual' ? document.getElementById('note').value.trim() : document.getElementById('qr-note').value.trim();
      if (!upiId) { showToast(' Please enter a UPI ID', 'error'); return; }
      if (!/^[\w.\-+]+@[\w]+$/.test(upiId)) { showToast(' Invalid UPI ID format', 'error'); return; }
      if (!amount || isNaN(parseFloat(amount)) || parseFloat(amount) <= 0) { showToast(' Please enter a valid amount', 'error'); return; }
      if (parseFloat(amount) > 100000) { showToast(' Amount cannot exceed 1,00,000', 'error'); return; }
      safeText(document.getElementById('cm-name'), name || upiId);
      safeText(document.getElementById('cm-upi'), upiId);
      document.getElementById('cm-amount').textContent = 'Rs ' + Number(amount).toLocaleString('en-IN');
      if (note) {
        safeText(document.getElementById('cm-note'), note);
        document.getElementById('cm-note-row').style.display = '';
      } else {
        document.getElementById('cm-note-row').style.display = 'none';
      }
      window._pendingData = { upiId, name, amount, note };
      document.getElementById('confirm-modal').classList.add('show');
    };
    window.closeModal = function() { document.getElementById('confirm-modal').classList.remove('show'); };
    window.handleModalBg = function(e) { if (e.target === e.currentTarget) closeModal(); };
    window.generateLink = function() {
      if (!requireAccess()) return;
      closeModal();
      const { upiId, name, amount, note } = window._pendingData;
      let link = `upi://pay?pa=${encodeURIComponent(upiId)}`;
      if (name) link += `&pn=${encodeURIComponent(name)}`;
      if (amount) link += `&am=${encodeURIComponent(amount)}`;
      link += `&cu=INR`;
      if (note) link += `&tn=${encodeURIComponent(note)}`;
      generatedLink = link;
      saveRecent(upiId, name);
      safeText(document.getElementById('out-name'), name || upiId);
      safeText(document.getElementById('out-upi'), upiId);
      document.getElementById('out-amount').textContent = 'Rs ' + Number(amount).toLocaleString('en-IN');
      safeText(document.getElementById('out-note'), note ? `"${note}"` : '');
      document.getElementById('link-display').textContent = link; // safe
      document.getElementById('pay-btn').href = link;
      const shareMsg = buildShareMessage(name || upiId, amount, note, link);
      document.getElementById('whatsapp-btn').href = `https://wa.me/?text=${encodeURIComponent(shareMsg)}`;
      document.getElementById('sms-btn').href = `sms:?body=${encodeURIComponent(shareMsg)}`;
      generateOutputQR(link);
      document.getElementById('input-section').style.display = 'none';
      document.getElementById('how-section').style.display = 'none';
      document.getElementById('output-section').style.display = 'block';
      document.getElementById('output-section').scrollIntoView({ behavior: 'smooth' });
      paymentDone = false;
      updateStatus(false);
      showToast(' Payment link generated!', 'success');
      vibrate([50,30,100]);
    };
    function buildShareMessage(name, amount, note, link) {
      return `Payment Request from ${name || 'a contact'}\n\nAmount: Rs ${Number(amount).toLocaleString('en-IN')}${note ? '\nNote: ' + note : ''}\n\n${link}\n\nJust click, UPI app opens. Enter PIN and done.`;
    }
    window.handlePayNow = function(e) {
      e.preventDefault();
      if (!requireAccess()) return;
      if (/Android|iPhone|iPad|iPod/i.test(navigator.userAgent)) window.location.href = generatedLink;
      else showToast(' Open this link on your phone to pay', 'success');
    };
    function generateOutputQR(link) {
      const container = document.getElementById('qr-output');
      container.innerHTML = '';
      try {
        new QRCode(container, { text: link, width: 200, height: 200, colorDark: '#0F172A', colorLight: '#FFFFFF', correctLevel: QRCode.CorrectLevel.M });
      } catch(e) {
        const img = document.createElement('img');
        img.src = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(link)}`;
        container.appendChild(img);
      }
    }
    window.copyLink = function() {
      if (!requireAccess()) return;
      navigator.clipboard.writeText(generatedLink).then(() => {
        const btn = document.getElementById('copy-btn');
        btn.textContent = ' Copied!'; btn.classList.add('copied');
        setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
        showToast(' Link copied!', 'success');
      }).catch(() => {
        const ta = document.createElement('textarea'); ta.value = generatedLink;
        document.body.appendChild(ta); ta.select(); document.execCommand('copy');
        document.body.removeChild(ta); showToast(' Link copied!', 'success');
      });
    };
    window.shareWhatsApp = function(e) { e.preventDefault(); if (!requireAccess()) return; window.open(document.getElementById('whatsapp-btn').href, '_blank', 'noopener,noreferrer'); vibrate([30]); };
    window.shareSMS = function(e) { e.preventDefault(); if (!requireAccess()) return; window.location.href = document.getElementById('sms-btn').href; };
    window.toggleStatus = function() {
      if (!requireAccess()) return;
      paymentDone = !paymentDone;
      updateStatus(paymentDone);
      showToast(paymentDone ? ' Payment marked as received!' : ' Marked as pending', '');
      if (paymentDone) vibrate([50,30,50,30,100]);
    };
    function updateStatus(done) {
      const dot = document.getElementById('status-dot');
      const text = document.getElementById('status-text');
      const sub = document.getElementById('status-sub');
      const btn = document.getElementById('status-btn');
      if (done) {
        dot.classList.add('done'); text.textContent = ' Payment Received';
        sub.textContent = 'Tap to mark as pending again'; btn.textContent = 'Mark Pending';
      } else {
        dot.classList.remove('done'); text.textContent = 'Awaiting Payment';
        sub.textContent = 'Mark as done when received'; btn.textContent = 'Mark Done';
      }
    }
    window.resetAll = function() {
      if (!requireAccess()) return;
      stopCameraScan();
      document.getElementById('output-section').style.display = 'none';
      document.getElementById('input-section').style.display = 'block';
      document.getElementById('how-section').style.display = 'block';
      ['upi-id','payee-name','amount','note'].forEach(id => {
        const el = document.getElementById(id); if (el) { el.value = ''; el.classList.remove('error','valid'); }
      });
      ['qr-name','qr-amount','qr-note'].forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
      scannedUPI = '';
      document.getElementById('qr-preview-container').style.display = 'none';
      document.getElementById('qr-scan-result').style.display = 'none';
      document.getElementById('qr-output').innerHTML = '';
      generatedLink = '';
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    let toastTimer;
    function showToast(msg, type = '') {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.className = 'toast ' + type + ' show';
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
    }
    function vibrate(pattern) { if (navigator.vibrate) navigator.vibrate(pattern); }

    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') closeModal();
      if (e.key === 'Enter' && document.getElementById('input-section').style.display !== 'none') handleGenerate();
    });

    authGate.start();
    renderRecent();
  })();



