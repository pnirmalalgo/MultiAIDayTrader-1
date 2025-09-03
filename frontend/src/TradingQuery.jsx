import { useState, useEffect } from "react";

export default function TradingQueryInput() {
  const [query, setQuery] = useState("");
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState("");
  const [files, setFiles] = useState([]);

  // Poll backend for task status
  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/task-status/${taskId}`);
        const data = await res.json();

        if (data.status === "SUCCESS") {
          setStatus("Completed! See generated charts below.");
          setFiles(data.files || []);
          clearInterval(interval);
        } else if (data.status === "FAILURE") {
          setStatus("Task failed. Check logs.");
          clearInterval(interval);
        } else {
          setStatus(`Task status: ${data.status}...`);
        }
      } catch (err) {
        setStatus("Error fetching task status.");
        console.error(err);
        clearInterval(interval);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [taskId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus("Submitting query...");
    setFiles([]);
    setTaskId(null);

    try {
      const res = await fetch("http://localhost:8000/api/submit-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      if (data.task_id) {
        setTaskId(data.task_id);
        setStatus("Task submitted. Waiting for results...");
      } else {
        setStatus(`Error: ${data.error || "No task_id returned."}`);
      }
    } catch (err) {
      setStatus("Error submitting query.");
      console.error(err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white shadow-lg rounded-2xl p-6 w-full max-w-4xl">
        <h1 className="text-2xl font-bold text-center mb-4">Trading Query</h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter trading query..."
            style={{
              width: "100%",
              height: "100px",   // 🔹 Reduced from 300px to 150px
              padding: "16px",
              borderRadius: "12px",
              border: "1px solid #ccc",
              resize: "none",
              outline: "none",
            }}
          />

          <button
            type="submit"
            style={{
              width: "100%",
              padding: "12px",
              marginTop: "8px",
              borderRadius: "12px",
              border: "none",
              backgroundColor: "#3b82f6",
              color: "white",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            Submit Query
          </button>
        </form>

        {status && (
          <div className="mt-4 p-3 bg-gray-100 border rounded-lg text-center">
            {status}
          </div>
        )}

        {files.length > 0 && (
          <div className="mt-6 p-3 bg-gray-50 border rounded-lg space-y-6">
            <h2 className="text-lg font-bold mb-2">Generated Charts:</h2>
            {files.map((file) => (
              <div key={file} className="space-y-2">
                <p className="font-semibold">{file}</p>
                <iframe
                  src={`http://localhost:8000/plots/${file}`}
                  title={file}
                  width="100%"
                  height="500px"
                  style={{ border: "1px solid #ddd", borderRadius: "12px" }}
                />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
