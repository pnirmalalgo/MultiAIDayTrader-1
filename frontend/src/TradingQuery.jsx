import { useState, useEffect } from "react";

export default function TradingQueryInput() {
  const [query, setQuery] = useState("");
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState("");
  const [files, setFiles] = useState([]);
  const [cot, setCot] = useState("");              // 🧠 Chain of Thought
  const [structuredJson, setStructuredJson] = useState(null);  // 📦 JSON output

  // Submit query: get CoT + JSON + enqueue execution
  const handleSubmit = async (e) => {
    e.preventDefault();
    setStatus("Interpreting query...");
    setFiles([]);
    setTaskId(null);
    setCot("");
    setStructuredJson(null);

    try {
      // 1️⃣ Call interpreter first
      const interpretRes = await fetch("http://localhost:8000/api/interpret-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const interpretData = await interpretRes.json();
      setCot(interpretData.thoughts);
      setStructuredJson(interpretData.structured_query);

      // 2️⃣ Now submit the original query for execution
      setStatus("Submitting for execution...");
      const execRes = await fetch("http://localhost:8000/api/submit-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const execData = await execRes.json();
      if (execData.task_id) {
        setTaskId(execData.task_id);
        setStatus("Task submitted. Waiting for results...");
      } else {
        setStatus(`Error: ${execData.error || "No task_id returned."}`);
      }

    } catch (err) {
      setStatus("Error interpreting or submitting query.");
      console.error(err);
    }
  };

  // Task polling...
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
              height: "100px",
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

        {/* ✅ Show Chain of Thought */}
        {cot && (
          <div className="mt-6 p-4 bg-gray-100 rounded-xl">
            <h3 className="text-lg font-semibold mb-2 text-blue-700">🧠 Chain of Thought</h3>
            <pre className="text-sm whitespace-pre-wrap">{cot}</pre>
          </div>
        )}

        {/* ✅ Show Structured JSON */}
        {structuredJson && (
          <div className="mt-4 p-4 bg-gray-100 rounded-xl">
            <h3 className="text-lg font-semibold mb-2 text-green-700">📦 Structured Query JSON</h3>
            <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(structuredJson, null, 2)}</pre>
          </div>
        )}

        {/* ✅ Status display */}
        {status && (
          <div className="mt-4 p-3 bg-gray-100 border rounded-lg text-center">
            {status}
          </div>
        )}

        {/* ✅ Plots */}
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
