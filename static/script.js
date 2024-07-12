const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white"; // Màu chữ trắng
let painting = false;

function startPosition(e) {
  painting = true;
  draw(e);
}

function endPosition() {
  painting = false;
  ctx.beginPath();
}

function draw(e) {
  if (!painting) return;
  ctx.lineWidth = 15;
  ctx.lineCap = "round";
  ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

canvas.addEventListener("mousedown", startPosition);
canvas.addEventListener("mouseup", endPosition);
canvas.addEventListener("mousemove", draw);

function clearCanvas() {
  ctx.fillRect(0, 0, canvas.width, canvas.height); // Reset canvas to black
  document.getElementById("single-result").style.display = "none"; // Ẩn kết quả khi xóa canvas
}

async function predict() {
  const image = canvas.toDataURL("image/png");
  try {
    const response = await fetch("/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image }),
    });

    const result = await response.json();
    const singleResultElement = document.getElementById("single-result");
    if (response.ok) {
      singleResultElement.innerText = `Predicted Label: ${result.label}`;
    } else {
      console.error("Error:", result.error);
      singleResultElement.innerText = `Error: ${result.error}`;
    }
    singleResultElement.style.display = "block"; // Hiển thị kết quả sau khi dự đoán
  } catch (error) {
    console.error("Error:", error);
    const singleResultElement = document.getElementById("single-result");
    singleResultElement.innerText = `Error: ${error}`;
    singleResultElement.style.display = "block"; // Hiển thị lỗi nếu có lỗi
  }
}

function uploadImages() {
  const fileInput = document.getElementById('upload');
  const files = fileInput.files;
  const imageContainer = document.getElementById('image-container');
  imageContainer.innerHTML = '';

  Array.from(files).forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = function(e) {
      const img = new Image();
      img.onload = async function() {
        // Resize and predict the uploaded image
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = 280;
        tempCanvas.height = 280;
        tempCtx.drawImage(img, 0, 0, 280, 280);

        const image = tempCanvas.toDataURL("image/png");
        try {
          const response = await fetch("/", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ image }),
          });

          const result = await response.json();
          if (response.ok) {
            // Display the uploaded image with predicted label
            const imgDisplay = document.createElement('div');
            imgDisplay.className = 'image-item';
            imgDisplay.innerHTML = `<img src="${e.target.result}" alt="Image ${index + 1}"><p>Predicted Label: ${result.label}</p><button class="delete-button" onclick="deleteImage(this)">X</button>`;
            imageContainer.appendChild(imgDisplay);
          } else {
            console.error("Error:", result.error);
            const imgDisplay = document.createElement('div');
            imgDisplay.className = 'image-item';
            imgDisplay.innerHTML = `<img src="${e.target.result}" alt="Image ${index + 1}"><p>Error: ${result.error}</p><button class="delete-button" onclick="deleteImage(this)">X</button>`;
            imageContainer.appendChild(imgDisplay);
          }
        } catch (error) {
          console.error("Error:", error);
          const imgDisplay = document.createElement('div');
          imgDisplay.className = 'image-item';
          imgDisplay.innerHTML = `<img src="${e.target.result}" alt="Image ${index + 1}"><p>Error: ${error}</p><button class="delete-button" onclick="deleteImage(this)">X</button>`;
          imageContainer.appendChild(imgDisplay);
        }
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  });
  document.getElementById('image-container').style.display = "block"; // Hiển thị container sau khi upload
}

function deleteImage(button) {
  const imageItem = button.parentElement;
  imageItem.remove();
  if (document.getElementById('image-container').children.length === 0) {
    document.getElementById('image-container').style.display = "none"; // Ẩn container nếu không còn ảnh nào
  }
}
