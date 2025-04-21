export default
class DrawingCanvas {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    this.isDrawing = false;
    
    this.initializeCanvas();
    this.setupEventListeners();
  }

  initializeCanvas() {
    this.ctx.fillStyle = 'black';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.lineWidth = 15;
    this.ctx.lineCap = 'round';
    this.ctx.strokeStyle = 'white';
  }

  setupEventListeners() {
    this.canvas.addEventListener('mousedown', this.startDrawing.bind(this));
    this.canvas.addEventListener('mousemove', this.draw.bind(this));
    this.canvas.addEventListener('mouseup', this.stopDrawing.bind(this));
    this.canvas.addEventListener('mouseout', this.stopDrawing.bind(this));
    
    // Touch support
    this.canvas.addEventListener('touchstart', this.handleTouch.bind(this));
    this.canvas.addEventListener('touchmove', this.handleTouch.bind(this));
    this.canvas.addEventListener('touchend', this.stopDrawing.bind(this));
  }

  handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(
      e.type === 'touchstart' ? 'mousedown' : 'mousemove',
      { clientX: touch.clientX, clientY: touch.clientY }
    );
    this.canvas.dispatchEvent(mouseEvent);
  }

  startDrawing(e) {
    this.isDrawing = true;
    this.draw(e);
  }

  draw(e) {
    if (!this.isDrawing) return;
    
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    this.ctx.lineTo(x, y);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(x, y);
  }

  stopDrawing() {
    this.isDrawing = false;
    this.ctx.beginPath();
  }

  clear() {
    this.ctx.fillStyle = 'black';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  getProcessedImage() {
    const { x, y, width, height } = this.getDrawingBoundingBox();
    
    if (width === 0 || height === 0) return new Array(784).fill(0);
    
    const scale = Math.min(20 / width, 20 / height);
    const newWidth = Math.floor(width * scale);
    const newHeight = Math.floor(height * scale);
    const offsetX = Math.floor((28 - newWidth) / 2);
    const offsetY = Math.floor((28 - newHeight) / 2);
    
    const resizeCanvas = document.createElement('canvas');
    resizeCanvas.width = 28;
    resizeCanvas.height = 28;
    const resizeCtx = resizeCanvas.getContext('2d');
    
    resizeCtx.fillStyle = 'black';
    resizeCtx.fillRect(0, 0, 28, 28);
    resizeCtx.drawImage(
      this.canvas,
      x, y, width, height,
      offsetX, offsetY, newWidth, newHeight
    );
    
    return this.extractPixelData(resizeCtx);
  }

  getDrawingBoundingBox() {
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const data = imageData.data;
    
    let minX = this.canvas.width, minY = this.canvas.height;
    let maxX = 0, maxY = 0;

    for (let y = 0; y < this.canvas.height; y++) {
      for (let x = 0; x < this.canvas.width; x++) {
        const idx = (y * this.canvas.width + x) * 4;
        if (data[idx] > 10) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
      }
    }
    
    const margin = 10;
    return {
      x: Math.max(0, minX - margin),
      y: Math.max(0, minY - margin),
      width: Math.min(this.canvas.width, maxX + margin) - Math.max(0, minX - margin),
      height: Math.min(this.canvas.height, maxY + margin) - Math.max(0, minY - margin)
    };
  }

  extractPixelData(ctx) {
    const imageData = ctx.getImageData(0, 0, 28, 28);
    const pixelArray = new Array(784);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      pixelArray[i / 4] = imageData.data[i] / 255;
    }
    
    return pixelArray;
  }
}