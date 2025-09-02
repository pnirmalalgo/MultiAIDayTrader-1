import { useState } from "react";

export default function TradingQueryInput() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [htmlFiles, setHtmlFiles] = useState([]);
  const [contextSummary, setContextSummary] = useState({ logs: [] });

  const handleSubmit = async (e) => {
    e.preventDefault();

    setStatus("Submitting query...");
    setHtmlFiles([]);
    setContextSummary({ input: query, logs: [] });

    try {
      const res = await fetch("http://localhost:8000/api/submit-query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();

      // Set final execution result directly
      setContextSummary(data);
      setStatus("Execution completed!");

      // Fetch generated HTML files
      fetchHtmlFiles();
    } catch (err) {
      console.error(err);
      setStatus("Error submitting query or executing task.");
    }
  };

  const fetchHtmlFiles = async () => {
    try {
      const res = await fetch("http://localhost:8000/api/list-html");
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setHtmlFiles(data.files || []);
    } catch (err) {
      console.error("Error fetching HTML files:", err);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white shadow-lg rounded-2xl p-6 w-full max-w-lg">
        <h1 className="text-2xl font-bold text-center mb-4">Trading Query</h1>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4 w-full">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter trading query..."
            style={{
              width: "100%",
              height: "200px",
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

        {contextSummary?.result && (
          <div className="mt-4 p-3 bg-gray-50 border rounded-lg">
            <h2 className="font-bold mb-2">Execution Summary:</h2>
            <pre>{JSON.stringify(contextSummary.result, null, 2)}</pre>
          </div>
        )}

        {contextSummary?.logs?.length > 0 && (
          <div className="mt-4 p-3 bg-gray-50 border rounded-lg">
            <h2 className="font-bold mb-2">Agent Logs (CoT / ReAct):</h2>
            <ul className="list-disc list-inside space-y-1 max-h-64 overflow-y-auto">
              {contextSummary.logs.map((log, index) => (
                <li key={index}>
                  <strong>{log.step}:</strong> {log.cot || log.react || JSON.stringify(log)}
                </li>
              ))}
            </ul>
          </div>
        )}

        {htmlFiles.length > 0 && (
          <div className="mt-6 p-3 bg-gray-50 border rounded-lg">
            <h2 className="text-lg font-bold mb-2">Generated HTML Files:</h2>
            <ul className="list-disc list-inside space-y-1">
              {htmlFiles.map((file) => (
                <li key={file}>
                  <a
                    href={`http://localhost:8000/api/html/${file}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 underline hover:text-blue-700"
                  >
                    {file}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
