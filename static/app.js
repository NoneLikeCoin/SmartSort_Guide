/* static/app.js */
/* 智能垃圾分类识别与投放指导系统 —— 前端逻辑
 * 流程：选择图片 → 本地校验 → POST /detect → 按比例画框 → POST /advise → 渲染建议
 * 安全约定：所有动态文本均通过 textContent 写入，不使用 innerHTML 拼接用户输入。
 */
(function () {
  "use strict";

  // ------------------------------------------------------------ 常量

  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_EXTS = ["jpg", "jpeg", "png"];
  const ALLOWED_MIME = ["image/jpeg", "image/png"];

  // 类别中文名（与服务端保持一致）
  const CLASS_LABELS = {
    cardboard: "纸板",
    glass: "玻璃",
    metal: "金属",
    paper: "纸",
    plastic: "塑料",
    trash: "其他垃圾",
  };

  // 不同类别的检测框颜色，便于区分
  const BOX_COLORS = {
    cardboard: "#b7791f",
    glass: "#3182ce",
    metal: "#718096",
    paper: "#2f855a",
    plastic: "#dd6b20",
    trash: "#c53030",
  };

  // ------------------------------------------------------------ DOM 引用

  const el = {
    form: document.getElementById("upload-form"),
    input: document.getElementById("image-input"),
    fileName: document.getElementById("file-name"),
    citySelect: document.getElementById("city-select"),
    sceneSelect: document.getElementById("scene-select"),
    submitBtn: document.getElementById("submit-btn"),
    errorBox: document.getElementById("error-box"),
    previewWrap: document.getElementById("preview-wrap"),
    previewImg: document.getElementById("preview-img"),
    previewEmpty: document.getElementById("preview-empty"),
    canvas: document.getElementById("overlay-canvas"),
    detectionList: document.getElementById("detection-list"),
    summary: document.getElementById("summary"),
    adviceList: document.getElementById("advice-list"),
  };

  // ------------------------------------------------------------ 状态

  let busy = false;
  let objectUrl = null;
  let lastResult = null; // { imageSize, detections }，用于窗口尺寸变化时重绘
  let resizeTimer = null;

  // ------------------------------------------------------------ 通用工具

  /** 人类可读的文件大小 */
  function formatSize(bytes) {
    if (bytes < 1024) {
      return bytes + " B";
    }
    if (bytes < 1024 * 1024) {
      return (bytes / 1024).toFixed(1) + " KB";
    }
    return (bytes / 1024 / 1024).toFixed(2) + " MB";
  }

  /** 展示错误信息（使用 textContent，避免注入） */
  function showError(message) {
    el.errorBox.textContent = String(message || "请求失败，请稍后重试。");
    el.errorBox.hidden = false;
  }

  /** 隐藏错误信息 */
  function clearError() {
    el.errorBox.textContent = "";
    el.errorBox.hidden = true;
  }

  /** 切换请求中的忙碌状态 */
  function setBusy(value) {
    busy = value;
    el.submitBtn.disabled = value;
    el.submitBtn.textContent = value ? "识别中…" : "开始识别";
  }

  /** 本地校验文件类型与大小，返回错误文案；通过时返回空串 */
  function validateFile(file) {
    if (!file) {
      return "请先选择一张图片。";
    }

    const name = file.name || "";
    const ext = name.indexOf(".") >= 0 ? name.split(".").pop().toLowerCase() : "";
    if (ALLOWED_EXTS.indexOf(ext) === -1) {
      return "仅支持 jpg、jpeg、png 格式的图片。";
    }

    if (file.type && ALLOWED_MIME.indexOf(file.type) === -1) {
      return "文件类型不是有效的图片，请重新选择。";
    }

    if (file.size === 0) {
      return "图片内容为空，请重新选择。";
    }

    if (file.size > MAX_FILE_SIZE) {
      return "图片大小超过 10MB 限制，请压缩后重试。";
    }

    return "";
  }

  // ------------------------------------------------------------ 网络请求

  /** 统一解析响应：非 2xx 时抛出带 error.message 的异常 */
  function parseResponse(response) {
    return response
      .json()
      .catch(function () {
        return null;
      })
      .then(function (data) {
        if (!response.ok) {
          const errorInfo = data && data.error ? data.error : {};
          const message =
            errorInfo.message || "请求失败（HTTP " + response.status + "）";
          const err = new Error(message);
          err.code = errorInfo.code || "UNKNOWN";
          throw err;
        }
        return data || {};
      });
  }

  /** POST /detect：上传图片并获取检测结果 */
  function requestDetect(file) {
    const formData = new FormData();
    formData.append("image", file, file.name);

    return fetch("/detect", {
      method: "POST",
      body: formData,
    }).then(parseResponse);
  }

  /** POST /advise：提交检测结果与城市/场景，获取投放建议 */
  function requestAdvise(detections) {
    const payload = {
      detections: detections.map(function (det) {
        return { class: det.class, confidence: det.confidence };
      }),
      city: el.citySelect.value,
      scene: el.sceneSelect.value,
    };

    return fetch("/advise", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(parseResponse);
  }

  // ------------------------------------------------------------ 画布绘制

  /** 清空画布 */
  function clearCanvas() {
    const ctx = el.canvas.getContext("2d");
    el.canvas.width = 0;
    el.canvas.height = 0;
    el.canvas.style.width = "0px";
    el.canvas.style.height = "0px";
    if (ctx) {
      ctx.clearRect(0, 0, 0, 0);
    }
  }

  /** 等待图片加载完成（已加载则立即返回） */
  function waitForImage(img) {
    return new Promise(function (resolve) {
      if (img.complete && img.naturalWidth) {
        resolve();
        return;
      }
      img.addEventListener("load", function () { resolve(); }, { once: true });
      img.addEventListener("error", function () { resolve(); }, { once: true });
    });
  }

  /** 按 image_size 与展示尺寸的比例，在图片上绘制检测框 */
  function drawBoxes(imageSize, detections) {
    if (!imageSize || !imageSize.width || !imageSize.height) {
      return;
    }

    const img = el.previewImg;
    const canvas = el.canvas;
    const displayWidth = img.clientWidth;
    const displayHeight = img.clientHeight;
    if (!displayWidth || !displayHeight) {
      return;
    }

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(displayWidth * dpr);
    canvas.height = Math.round(displayHeight * dpr);
    canvas.style.width = displayWidth + "px";
    canvas.style.height = displayHeight + "px";

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, displayWidth, displayHeight);

    const scaleX = displayWidth / imageSize.width;
    const scaleY = displayHeight / imageSize.height;

    (detections || []).forEach(function (det) {
      const box = det && Array.isArray(det.box) ? det.box : null;
      if (!box || box.length !== 4) {
        return;
      }

      const x = box[0] * scaleX;
      const y = box[1] * scaleY;
      const w = Math.max((box[2] - box[0]) * scaleX, 1);
      const h = Math.max((box[3] - box[1]) * scaleY, 1);
      const color = BOX_COLORS[det.class] || "#2f855a";

      // 检测框
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      // 类别标签
      const labelName = CLASS_LABELS[det.class] || det.class || "";
      const confidence = typeof det.confidence === "number" ? det.confidence : 0;
      const label = labelName + " " + Math.round(confidence * 100) + "%";

      ctx.font = '600 13px -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif';
      const padding = 6;
      const textWidth = ctx.measureText(label).width;
      const labelHeight = 20;
      const labelX = x;
      const labelY = y - labelHeight - 2 >= 0 ? y - labelHeight - 2 : y + 2;

      ctx.fillStyle = color;
      ctx.fillRect(labelX, labelY, textWidth + padding * 2, labelHeight);
      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "middle";
      ctx.fillText(label, labelX + padding, labelY + labelHeight / 2);
    });
  }

  /** 窗口尺寸变化时按新比例重绘 */
  function redraw() {
    if (!lastResult) {
      return;
    }
    drawBoxes(lastResult.imageSize, lastResult.detections);
  }

  // ------------------------------------------------------------ 结果渲染

  /** 渲染检测结果列表 */
  function renderDetections(data) {
    el.detectionList.replaceChildren();

    const detections = Array.isArray(data.detections) ? data.detections : [];
    detections.forEach(function (det) {
      const li = document.createElement("li");
      li.className = "detection-item";

      const name = document.createElement("span");
      name.className = "detection-item__name";
      name.textContent = CLASS_LABELS[det.class] || det.class || "未知";

      const confidence = typeof det.confidence === "number" ? det.confidence : 0;
      const conf = document.createElement("span");
      conf.className = "detection-item__conf";
      conf.textContent = "置信度 " + (confidence * 100).toFixed(1) + "%";

      li.appendChild(name);
      li.appendChild(conf);
      el.detectionList.appendChild(li);
    });
  }

  /** 渲染投放指导列表与汇总说明 */
  function renderAdvice(data) {
    el.adviceList.replaceChildren();

    el.summary.textContent =
      typeof data.summary === "string" ? data.summary : "";

    const items = Array.isArray(data.items) ? data.items : [];
    items.forEach(function (item) {
      const li = document.createElement("li");
      li.className = "advice-item";

      const head = document.createElement("div");
      head.className = "advice-item__head";

      const name = document.createElement("span");
      name.className = "advice-item__class";
      name.textContent = CLASS_LABELS[item.class] || item.class || "未知";

      const category = document.createElement("span");
      category.className = "advice-item__category";
      category.textContent = item.category || "—";

      head.appendChild(name);
      head.appendChild(category);

      const bin = document.createElement("div");
      bin.className = "advice-item__bin";
      bin.textContent = "投放容器：" + (item.bin || "—");

      const reason = document.createElement("p");
      reason.className = "advice-item__reason";
      reason.textContent = item.reason || "";

      li.appendChild(head);
      li.appendChild(bin);
      li.appendChild(reason);
      el.adviceList.appendChild(li);
    });
  }

  /** 清空上次的检测与建议结果 */
  function resetResults() {
    lastResult = null;
    el.detectionList.replaceChildren();
    el.adviceList.replaceChildren();
    el.summary.textContent = "";
    clearCanvas();
  }

  // ------------------------------------------------------------ 事件处理

  /** 选择文件后：本地预览 + 重置结果 */
  function handleFileChange() {
    clearError();
    resetResults();

    const file = el.input.files && el.input.files[0];

    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }

    if (!file) {
      el.fileName.textContent = "未选择文件";
      el.previewImg.hidden = true;
      el.previewEmpty.hidden = false;
      return;
    }

    el.fileName.textContent = file.name + "（" + formatSize(file.size) + "）";

    objectUrl = URL.createObjectURL(file);
    el.previewImg.hidden = false;
    el.previewEmpty.hidden = true;
    el.previewImg.src = objectUrl;
    el.previewImg.onload = function () {
      clearCanvas();
    };
  }

  /** 提交表单：检测 → 画框 → 建议 */
  function handleSubmit(event) {
    event.preventDefault();

    if (busy) {
      return;
    }

    clearError();
    resetResults();

    const file = el.input.files && el.input.files[0];
    const message = validateFile(file);
    if (message) {
      showError(message);
      return;
    }

    setBusy(true);

    requestDetect(file)
      .then(function (detectData) {
        renderDetections(detectData);
        lastResult = {
          imageSize: detectData.image_size,
          detections: detectData.detections || [],
        };

        // 等图片渲染完成后再按比例画框，避免尺寸为 0
        return waitForImage(el.previewImg).then(function () {
          drawBoxes(lastResult.imageSize, lastResult.detections);
          return requestAdvise(lastResult.detections);
        });
      })
      .then(function (adviceData) {
        renderAdvice(adviceData);
      })
      .catch(function (err) {
        showError(err && err.message ? err.message : "请求失败，请稍后重试。");
      })
      .finally(function () {
        setBusy(false);
      });
  }

  // ------------------------------------------------------------ 初始化

  el.input.addEventListener("change", handleFileChange);
  el.form.addEventListener("submit", handleSubmit);

  window.addEventListener("resize", function () {
    if (resizeTimer) {
      clearTimeout(resizeTimer);
    }
    resizeTimer = setTimeout(redraw, 120);
  });
})();