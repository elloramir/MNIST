import LogisticRegressionModel from "./LogisticRegressionModel.js"
import DrawingCanvas from "./DrawingCanvas.js"
import ModelPersistence from "./ModelPersistence.js"
 
export default
class MNISTViewController {
  constructor() {
    this.model = new LogisticRegressionModel();
    this.drawingCanvas = new DrawingCanvas(document.getElementById('drawingCanvas'));
    
    this.uiElements = {
      trainBtn: document.getElementById('trainBtn'),
      downloadBtn: document.getElementById('downloadBtn'),
      loadBtn: document.getElementById('loadBtn'),
      weightsFile: document.getElementById('weightsFile'),
      statusDiv: document.getElementById('status'),
      resultsDiv: document.getElementById('results'),
      predictBtn: document.getElementById('predictBtn'),
      clearBtn: document.getElementById('clearBtn'),
      predictionResult: document.getElementById('predictionResult')
    };
    
    this.model.updateStatus = this.updateStatus.bind(this);
    this.setupEventListeners();
    this.updateStatus("Pronto para treinar o modelo ou carregar pesos existentes.");
  }

  setupEventListeners() {
    this.uiElements.trainBtn.addEventListener('click', this.handleTrainClick.bind(this));
    this.uiElements.downloadBtn.addEventListener('click', this.handleDownloadClick.bind(this));
    this.uiElements.loadBtn.addEventListener('click', this.handleLoadClick.bind(this));
    this.uiElements.predictBtn.addEventListener('click', this.handlePredictClick.bind(this));
    this.uiElements.clearBtn.addEventListener('click', this.handleClearClick.bind(this));
    this.uiElements.weightsFile.addEventListener('change', () => {
      this.uiElements.loadBtn.disabled = !this.uiElements.weightsFile.files.length;
    });
  }

  async handleTrainClick() {
    this.setButtonState(true);
    this.updateStatus("Iniciando treinamento...");
    
    try {
      const { train, test } = await DataProcessor.loadDataset("data/train.gz", "data/test.gz");
      const flatTrain = DataProcessor.flattenDataset(train);
      const flatTest = DataProcessor.flattenDataset(test);
      
      const { weights, biases } = this.model.train(flatTrain);
      const accuracy = this.model.evaluate(flatTest) * 100;
      
      this.uiElements.resultsDiv.textContent = 
        `Modelo treinado com sucesso! Acurácia no teste: ${accuracy.toFixed(2)}%`;
      this.uiElements.downloadBtn.disabled = false;
      this.uiElements.predictBtn.disabled = false;
    } catch (error) {
      this.updateStatus("Erro durante o treinamento: " + error.message);
      } finally {
        this.setButtonState(false);
      }
  }

  handleDownloadClick() {
    if (!this.model.trained) return;
    
    ModelPersistence.downloadWeights(this.model.weights, this.model.biases);
    this.updateStatus("Pesos do modelo baixados com sucesso!");
  }

  async handleLoadClick() {
    if (!this.uiElements.weightsFile.files.length) return;
    
    this.uiElements.loadBtn.disabled = true;
    this.updateStatus("Carregando pesos do modelo...");
    
    try {
      const modelData = await ModelPersistence.loadWeightsFromFile(this.uiElements.weightsFile.files[0]);
      this.model.weights = modelData.weights;
      this.model.biases = modelData.biases;
      this.model.trained = true;
      
      this.updateStatus("Pesos do modelo carregados com sucesso!");
      this.uiElements.resultsDiv.textContent = "Modelo pronto para inferência!";
      this.uiElements.predictBtn.disabled = false;
    } catch (error) {
      this.updateStatus("Erro ao carregar pesos: " + error.message);
    } finally {
      this.uiElements.loadBtn.disabled = false;
    }
  }

  handlePredictClick() {
    if (!this.model.trained) {
      this.uiElements.predictionResult.textContent = "Modelo não treinado ou carregado ainda.";
      return;
    }
    
    const pixelArray = this.drawingCanvas.getProcessedImage();
    const { predicted, probs } = this.model.predict(pixelArray);
    
    let resultHTML = `<h3>Predição: ${predicted}</h3><h4>Probabilidades:</h4><ul>`;
    probs.forEach((prob, i) => {
      resultHTML += `<li>${i}: ${(prob * 100).toFixed(2)}%</li>`;
    });
    
    this.uiElements.predictionResult.innerHTML = resultHTML + "</ul>";
  }

  handleClearClick() {
    this.drawingCanvas.clear();
    this.uiElements.predictionResult.textContent = '';
  }

  updateStatus(message) {
    this.uiElements.statusDiv.textContent = message;
  }

  setButtonState(training) {
    this.uiElements.trainBtn.disabled = training;
    this.uiElements.downloadBtn.disabled = training;
  }
}