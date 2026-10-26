import { Upload } from "lucide-react";

export default function BatchUploadPanel({
  onFileSelect,
  onReset,
  fileName
}) {
  function handleFile(event) {
    const file = event.target.files?.[0];

    if (file) {
      onFileSelect(file);
    }
  }

  function handleDrop(event) {
    event.preventDefault();

    const file = event.dataTransfer.files?.[0];

    if (file) {
      onFileSelect(file);
    }
  }

  return (
    <div className="card">
      <h3>CSV Upload</h3>

      <div
        className="upload-zone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <Upload size={34} />

        <p>
          Drag & drop a CSV file or choose one
          below
        </p>

        <input
          type="file"
          accept=".csv"
          onChange={handleFile}
        />
      </div>

      {fileName && (
        <p className="file-name">
          Loaded: {fileName}
        </p>
      )}

      <button
        className="secondary-btn"
        onClick={onReset}
      >
        Reset
      </button>
    </div>
  );
}