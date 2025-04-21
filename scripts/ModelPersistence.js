export default
class ModelPersistence {
  static downloadWeights(weights, biases) {
    const modelData = { weights, biases };
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(modelData));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "mnist_weights.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    document.body.removeChild(downloadAnchor);
  }

  static loadWeightsFromFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = event => resolve(JSON.parse(event.target.result));
      reader.onerror = error => reject(error);
      reader.readAsText(file);
    });
  }
}