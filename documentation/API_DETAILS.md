# Sales Rep RAG Agent — API Reference

Use this document to build a frontend (e.g. React) that talks to the FastAPI backend.

---

## Base URL and server

- **Default base URL:** `http://localhost:8000`
- **Start the server:** From the project root, run:
  ```bash
  uvicorn api:app --reload --host 0.0.0.0 --port 8000
  ```
- **CORS:** The API allows `http://localhost:3000` and `http://127.0.0.1:3000` by default. For other origins, set `API_CORS_ORIGINS` (comma-separated) in your environment or `.env`.

---

## Endpoints

### 1. Health check

**GET** `/health` or **GET** `/api/health`

Check that the API is up.

**Response:** `200 OK`

```json
{
  "status": "ok"
}
```

**Example (fetch):**
```javascript
const res = await fetch('http://localhost:8000/api/health');
const data = await res.json(); // { status: "ok" }
```

---

### 2. Run sales-rep flow (streaming)

**POST** `/api/run`

Starts a full sales-rep agent run: RAG index is built, then the LangGraph loop runs (Reason → Decide → Act → Observe → Update → Reflect) until the agent stops or hits `max_steps`. The response is a **Server-Sent Events (SSE)** stream. Each event is a JSON object with a `type` field; the stream ends with a `done` event.

**Request**

- **Content-Type:** `application/json`
- **Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `my_company_description` | string | Yes | Who you represent (your company / product summary). |
| `prospect_company_name` | string | Yes | Prospect company name. |
| `prospect_industry` | string | Yes | Prospect industry. |
| `prospect_profile_text` | string | Yes | Prospect profile or website text. |
| `max_steps` | number | No | Max agent steps (default from server config, e.g. 15). |
| `run_initial_search` | boolean | No | If `true`, run an initial web search before the loop (default `true`). |

**Example request body:**
```json
{
  "my_company_description": "K2X Technologies: AI-driven software for industrial efficiency and downtime reduction.",
  "prospect_company_name": "Acme Corp",
  "prospect_industry": "Manufacturing",
  "prospect_profile_text": "Acme Corp manufactures precision parts. 500 employees. HQ in Chicago.",
  "max_steps": 8,
  "run_initial_search": true
}
```

**Response**

- **Status:** `200 OK`
- **Content-Type:** `text/event-stream`
- **Body:** SSE stream. Each event line is `data: <JSON>\n\n`. Parse the JSON to get the event object.

**Event types and payloads**

Every event is a JSON object. Use the `type` field to branch your UI.

---

#### `start`

Emitted once when the run has started (after RAG index is built).

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"start"` |
| `prospect` | string | Prospect company name. |
| `industry` | string | Prospect industry. |
| `company_summary` | string | Truncated seller summary (~100 chars). |

**Example:**
```json
{
  "type": "start",
  "prospect": "Acme Corp",
  "industry": "Manufacturing",
  "company_summary": "K2X Technologies: AI-driven software for industrial efficiency and downtime reduction."
}
```

---

#### `log`

Emitted for each log line from the agent (reasoning, retrieval, search, etc.). Message is truncated to 500 characters.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"log"` |
| `level` | string | Log level, e.g. `"INFO"`, `"DEBUG"`, `"WARNING"`. |
| `message` | string | Log message. |
| `logger` | string | Logger name, e.g. `"sales_rep.agent"`, `"sales_rep.rag"`. |

**Example:**
```json
{
  "type": "log",
  "level": "INFO",
  "message": "=== REASON NODE === ...",
  "logger": "sales_rep.agent"
}
```

---

#### `step`

Emitted for each graph node execution (reason, decide, act, observe, update, reflect, generate_answer).

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"step"` |
| `node` | string | One of: `"reason"`, `"decide"`, `"act"`, `"observe"`, `"update"`, `"reflect"`, `"generate_answer"`. |
| `step_number` | number | Current step index (1-based). |
| `updates` | object | Sanitised state updates for this node (truncated). May include `reason_text`, `decision` (tool_id, should_stop, confidence), `observation`, `final_response`, etc. |

**Example:**
```json
{
  "type": "step",
  "node": "decide",
  "step_number": 1,
  "updates": {
    "decision": {
      "tool_id": "search_web",
      "should_stop": false,
      "confidence": 0.5
    }
  }
}
```

---

#### `tool`

Emitted after a tool is executed (search_web, extract_insights, consult_oracle). Result is truncated to 400 characters.

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"tool"` |
| `tool_id` | string | `"search_web"`, `"extract_insights"`, or `"consult_oracle"`. |
| `input` | object | Tool input, e.g. `{"query": "Acme Corp manufacturing"}` for search_web. |
| `result_preview` | string | Truncated tool result or observation. |

**Example (search):**
```json
{
  "type": "tool",
  "tool_id": "search_web",
  "input": { "query": "Acme Corp manufacturing" },
  "result_preview": "[1] Acme Corp - Wikipedia\nURL: https://...\nAcme Corp is a..."
}
```

**Example (oracle):**
```json
{
  "type": "tool",
  "tool_id": "consult_oracle",
  "input": {
    "question": "What is the strongest value hypothesis for selling K2X to Acme?",
    "context": "Acme Corp manufactures..."
  },
  "result_preview": "The strongest value hypothesis is reducing unplanned downtime..."
}
```

---

#### `done`

Emitted exactly once at the end of the stream (success or failure). After this event, close the connection.

**On success:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"done"` |
| `result` | object | Parsed answer with three keys (see below). |
| `tools_used` | array of strings | List of tool IDs used this run, e.g. `["search_web", "extract_insights", "consult_oracle"]`. |
| `steps` | number | Total step count. |

`result` shape:

| Field | Type | Description |
|-------|------|-------------|
| `value_hypothesis` | string | Value hypothesis text. |
| `messaging_angle` | string | Messaging angle text. |
| `supporting_evidence` | string | Supporting evidence text. |

**Example (success):**
```json
{
  "type": "done",
  "result": {
    "value_hypothesis": "K2X can help Acme reduce unplanned downtime by 20% through predictive maintenance.",
    "messaging_angle": "Acme, as a precision manufacturer, loses revenue every hour of downtime—we help you predict and prevent it.",
    "supporting_evidence": "- Acme has 500 employees and likely multiple production lines.\n- Industry trend: manufacturers adopting AI for OEE."
  },
  "tools_used": ["extract_insights", "search_web", "consult_oracle"],
  "steps": 4
}
```

**On failure:**

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `"done"` |
| `error` | string | Error message. |
| `result` | null | Always `null` on error. |

**Example (failure):**
```json
{
  "type": "done",
  "error": "Connection refused to Ollama",
  "result": null
}
```

---

## Consuming the stream in your app

### Using `fetch` + ReadableStream

```javascript
async function runAgent(payload) {
  const res = await fetch('http://localhost:8000/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        switch (data.type) {
          case 'start':
            console.log('Run started', data.prospect, data.industry);
            break;
          case 'log':
            console.log(`[${data.level}] ${data.message}`);
            break;
          case 'step':
            console.log('Step', data.node, data.step_number, data.updates);
            break;
          case 'tool':
            console.log('Tool', data.tool_id, data.input, data.result_preview);
            break;
          case 'done':
            if (data.error) console.error(data.error);
            else console.log('Result', data.result, 'tools_used', data.tools_used);
            return data;
        }
      }
    }
  }
  return null;
}

// Usage
const result = await runAgent({
  my_company_description: 'Your company description',
  prospect_company_name: 'Acme',
  prospect_industry: 'Manufacturing',
  prospect_profile_text: 'Acme Corp...',
});
```

### Using EventSource (if you prefer)

`EventSource` is for GET requests; for POST you cannot use it directly. Use `fetch` + stream as above, or a library that supports POST + SSE (e.g. `@microsoft/fetch-event-source` or similar).

---

## Summary table

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` or `/api/health` | Health check. |
| POST | `/api/run` | Run sales-rep flow; SSE stream (start, log, step, tool, done). |
| POST | `/api/find-prospects` | Find prospect companies; SSE stream (start, log, step, tool, done). |

---

## Notes

- **Timeouts:** A run can take 1–5+ minutes depending on steps and tools. Ensure your HTTP client and any proxies allow long-lived responses.
- **Log files:** The server writes full run logs to `logs/run_YYYYMMDD_HHMMSS.txt` and prospect logs to `logs/prospect_run_YYYYMMDD_HHMMSS.txt`; the SSE stream is a subset for the UI.
- **Truncation:** Log messages, tool result previews, and step updates are truncated on the server to keep the stream manageable; full detail is in the run log file.

---

## Prospect Finder endpoint

### POST `/api/find-prospects`

Starts a Prospect Finder agent run. Given your company description, a location, and a target count, the agent autonomously searches LinkedIn (via DuckDuckGo `site:linkedin.com/company` queries) and enriches each discovered company. Falls back to plain web search if LinkedIn is inaccessible.

**Request**

- **Content-Type:** `application/json`
- **Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `my_company_description` | string | Yes | Who you represent (your company / product summary). |
| `location` | string | Yes | City, region, or country to find prospects in, e.g. `"Islamabad, Pakistan"`. |
| `num_prospects` | number | Yes | How many prospect companies to find. |
| `industry_focus` | string | No | Optional industry or sector to target, e.g. `"IT services"`, `"manufacturing"`. |
| `max_steps` | number | No | Max agent steps (default from server config). |

**Example request body:**
```json
{
  "my_company_description": "K2X Technologies: AI-driven software for industrial efficiency.",
  "location": "Islamabad, Pakistan",
  "num_prospects": 5,
  "industry_focus": "IT services"
}
```

**Response**

SSE stream — same protocol as `/api/run`. Each event is `data: <JSON>\n\n`.

**Event types**

The `start`, `log`, `step`, and `tool` events have the same shape as for `/api/run`. The `done` event has a different `result`:

#### `start`

```json
{
  "type": "start",
  "location": "Islamabad, Pakistan",
  "num_prospects": 5,
  "industry_focus": "IT services",
  "company_summary": "K2X Technologies: AI-driven software for industrial efficiency."
}
```

#### `tool` — tool IDs for prospect finder

| `tool_id` | What it does |
|-----------|--------------|
| `search_linkedin_companies` | DuckDuckGo `site:linkedin.com/company` query; auto-falls back to plain web |
| `search_web_general` | Plain DuckDuckGo fallback for company lists or directories |
| `fetch_company_info` | Enriches a discovered company name with website/industry/size |
| `consult_oracle` | GPT-4o-mini — optional, used when stuck or for final list synthesis |

**Example:**
```json
{
  "type": "tool",
  "tool_id": "search_linkedin_companies",
  "input": { "location": "Islamabad, Pakistan", "industry_hint": "IT services" },
  "result_preview": "[1] Systems Limited\nURL: https://linkedin.com/company/systems-limited\n..."
}
```

#### `done` — on success

`result` is an object with a `prospects` array:

```json
{
  "type": "done",
  "result": {
    "prospects": [
      {
        "company_name": "Systems Limited",
        "linkedin_url": "https://www.linkedin.com/company/systems-limited",
        "website": "https://www.systemsltd.com",
        "industry": "IT Services",
        "size_estimate": "1000-5000 employees",
        "location": "Islamabad, Pakistan",
        "key_facts": [
          "One of Pakistan's largest IT companies",
          "Listed on Pakistan Stock Exchange"
        ]
      }
    ]
  },
  "tools_used": ["search_linkedin_companies", "fetch_company_info"],
  "steps": 4
}
```

**Prospect object fields:**

| Field | Type | Description |
|-------|------|-------------|
| `company_name` | string | Company name. |
| `linkedin_url` | string or null | LinkedIn company page URL. |
| `website` | string or null | Company website. |
| `industry` | string or null | Industry or sector. |
| `size_estimate` | string or null | Estimated employee count, e.g. `"50-200 employees"`. |
| `location` | string or null | Company location. |
| `key_facts` | array of strings | Short bullet facts from research. |

#### `done` — on failure

```json
{
  "type": "done",
  "error": "Connection refused to Ollama",
  "result": null
}
```

**JavaScript example:**

```javascript
async function findProspects(payload) {
  const res = await fetch('http://localhost:8000/api/find-prospects', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop() || '';
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.type === 'done') {
          if (data.error) console.error(data.error);
          else console.log('Prospects:', data.result.prospects);
          return data;
        }
        if (data.type === 'tool') {
          console.log('Tool called:', data.tool_id, data.input);
        }
        if (data.type === 'step') {
          const found = data.updates?.found_companies?.length || 0;
          console.log(`Step ${data.step_number} [${data.node}] — found: ${found}`);
        }
      }
    }
  }
  return null;
}

// Usage
const result = await findProspects({
  my_company_description: 'K2X Technologies: AI-driven industrial software.',
  location: 'Islamabad, Pakistan',
  num_prospects: 5,
  industry_focus: 'IT services',
});
```
