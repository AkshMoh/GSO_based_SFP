import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [summary, setSummary] = useState("");
  const [error, setError] = useState("");
  const [model, setModel] = useState("QDA");
  const [iterations, setIterations] = useState(10);

  // =========================
  // FILE UPLOAD HANDLER
  // =========================
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    if (!selectedFile) return;

    // Accept CSV / XLS / XLSX
    const allowedExtensions = ["csv", "xls", "xlsx"];

    const extension = selectedFile.name
      .split(".")
      .pop()
      .toLowerCase();

    // ALSO allow files with no visible extension
    // because Windows sometimes hides it
    if (
      !allowedExtensions.includes(extension) &&
      selectedFile.type !== "application/vnd.ms-excel"
    ) {
      setError(
        "❌ Please upload a CSV, XLS, or XLSX dataset file."
      );
      return;
    }

    setFile(selectedFile);
    setFileName(selectedFile.name);
    setResults(null);
    setSummary("");
    setError("");
  };

  // =========================
  // PREDICTION FUNCTION
  // =========================
  const handlePredict = async () => {
    if (!file) {
      setError("❌ Please upload a dataset first!");
      return;
    }

    setLoading(true);
    setError("");
    setResults(null);
    setSummary("");

    try {
      // Dynamic import
      const { Client } = await import("@gradio/client");

      // Connect to HF Space
      const client = await Client.connect(
        "Akshat-22-Mohit/gso-fault-predictor"
      );

      // Predict
      const response = await client.predict(
        "/predict_faults",
        [
          file,
          model,
          iterations,
        ]
      );

      const data = response.data;

      setResults(data[0]);
      setSummary(data[1]);

    } catch (err) {
      console.error(err);

      setError(
        "❌ Error: " +
        (err?.message || "Prediction failed")
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: "1000px",
        margin: "0 auto",
        padding: "20px",
        fontFamily: "Arial",
      }}
    >
      {/* ========================= */}
      {/* HEADER */}
      {/* ========================= */}
      <h1
        style={{
          color: "#2c3e50",
          borderBottom: "3px solid #e74c3c",
          paddingBottom: "10px",
        }}
      >
        🐍 GSO-Based Software Fault Predictor
      </h1>

      <p
        style={{
          color: "#7f8c8d",
          fontSize: "18px",
        }}
      >
        Upload any software metrics dataset.
        GSO will automatically select optimal features
        and predict fault-prone modules.
      </p>

      {/* ========================= */}
      {/* UPLOAD SECTION */}
      {/* ========================= */}
      <div
        style={{
          background: "#f8f9fa",
          padding: "25px",
          borderRadius: "12px",
          marginBottom: "20px",
          boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
        }}
      >
        <h2>📂 Upload Dataset File</h2>

        <input
          type="file"
          accept=".csv,.xls,.xlsx"
          onChange={handleFileChange}
          style={{
            marginBottom: "10px",
            display: "block",
            padding: "10px",
          }}
        />

        {fileName && (
          <p
            style={{
              color: "#27ae60",
              fontWeight: "bold",
            }}
          >
            ✅ Selected: {fileName}
          </p>
        )}

        {/* ========================= */}
        {/* OPTIONS */}
        {/* ========================= */}
        <div
          style={{
            display: "flex",
            gap: "30px",
            marginTop: "20px",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          {/* MODEL */}
          <div>
            <label>
              <b>🤖 Classifier:</b>
            </label>

            <select
              value={model}
              onChange={(e) =>
                setModel(e.target.value)
              }
              style={{
                marginLeft: "10px",
                padding: "8px",
                borderRadius: "5px",
              }}
            >
              <option value="QDA">QDA</option>
              <option value="KNN">KNN</option>
              <option value="NB">NB</option>
              <option value="RF">RF</option>
            </select>
          </div>

          {/* ITERATIONS */}
          <div>
            <label>
              <b>
                ⚙️ GSO Iterations: {iterations}
              </b>
            </label>

            <input
              type="range"
              min="5"
              max="30"
              step="5"
              value={iterations}
              onChange={(e) =>
                setIterations(
                  Number(e.target.value)
                )
              }
              style={{
                marginLeft: "15px",
              }}
            />
          </div>
        </div>
      </div>

      {/* ========================= */}
      {/* BUTTON */}
      {/* ========================= */}
      <button
        onClick={handlePredict}
        disabled={loading || !file}
        style={{
          width: "100%",
          padding: "16px",
          background: loading
            ? "#95a5a6"
            : "#e74c3c",
          color: "white",
          border: "none",
          borderRadius: "10px",
          fontSize: "20px",
          fontWeight: "bold",
          cursor: loading
            ? "not-allowed"
            : "pointer",
          marginBottom: "25px",
        }}
      >
        {loading
          ? "⏳ Running GSO + Predicting... Please Wait"
          : "🔍 Run GSO + Predict"}
      </button>

      {/* ========================= */}
      {/* ERROR */}
      {/* ========================= */}
      {error && (
        <div
          style={{
            background: "#fadbd8",
            padding: "15px",
            borderRadius: "10px",
            marginBottom: "20px",
            color: "#c0392b",
            fontWeight: "bold",
          }}
        >
          {error}
        </div>
      )}

      {/* ========================= */}
      {/* SUMMARY */}
      {/* ========================= */}
      {summary && (
        <div
          style={{
            background: "#eafaf1",
            padding: "20px",
            borderRadius: "10px",
            marginBottom: "20px",
          }}
        >
          <h2>📋 Summary & Metrics</h2>

          <pre
            style={{
              whiteSpace: "pre-wrap",
              fontFamily: "monospace",
              fontSize: "14px",
            }}
          >
            {summary}
          </pre>
        </div>
      )}

      {/* ========================= */}
      {/* RESULTS TABLE */}
      {/* ========================= */}
      {results &&
        results.data &&
        results.headers && (
          <div
            style={{
              overflowX: "auto",
            }}
          >
            <h2>📊 Module-wise Predictions</h2>

            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "13px",
              }}
            >
              <thead>
                <tr
                  style={{
                    background: "#2c3e50",
                    color: "white",
                  }}
                >
                  {results.headers.map(
                    (header, index) => (
                      <th
                        key={index}
                        style={{
                          padding: "10px",
                          textAlign: "left",
                        }}
                      >
                        {header}
                      </th>
                    )
                  )}
                </tr>
              </thead>

              <tbody>
                {results.data
                  .slice(0, 50)
                  .map((row, rowIndex) => (
                    <tr
                      key={rowIndex}
                      style={{
                        background:
                          rowIndex % 2 === 0
                            ? "#f8f9fa"
                            : "white",
                      }}
                    >
                      {row.map(
                        (cell, cellIndex) => (
                          <td
                            key={cellIndex}
                            style={{
                              padding: "8px",
                              borderBottom:
                                "1px solid #dee2e6",
                            }}
                          >
                            {cell}
                          </td>
                        )
                      )}
                    </tr>
                  ))}
              </tbody>
            </table>

            {results.data.length > 50 && (
              <p
                style={{
                  color: "#7f8c8d",
                  textAlign: "center",
                  marginTop: "10px",
                }}
              >
                Showing first 50 of{" "}
                {results.data.length} modules
              </p>
            )}
          </div>
        )}

      {/* ========================= */}
      {/* FOOTER */}
      {/* ========================= */}
      <div
        style={{
          marginTop: "40px",
          textAlign: "center",
          color: "#7f8c8d",
          borderTop: "1px solid #dee2e6",
          paddingTop: "20px",
        }}
      >
        <p>
          Algorithm: Glider Snake Optimization
          (GSO)
        </p>

        <p>
          Supported Formats: CSV / XLS / XLSX
        </p>

        <p>
          Models: QDA | KNN | NB | RF
        </p>
      </div>
    </div>
  );
}

export default App;