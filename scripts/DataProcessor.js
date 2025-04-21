export default
class DataProcessor {
  static async loadDataset(trainPath, testPath) {
    const [train, test] = await Promise.all([
      DataProcessor.downloadAndParseGzip(trainPath),
      DataProcessor.downloadAndParseGzip(testPath)
    ]);
    return { train, test };
  }

  static async downloadAndParseGzip(url) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const data = new Uint8Array(buffer);
    const inflated = pako.inflate(data, { to: "string" });
    return JSON.parse(inflated);
  }

  static flattenDataset(dataset) {
    return dataset.data.flatMap(item => {
      const label = parseInt(item.label, 10);
      return item.images.map(image => ({
        x: image.map(pixel => pixel / 255),
        label
      }));
    });
  }
}