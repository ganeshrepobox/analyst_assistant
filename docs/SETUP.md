# Setup

Two credentials are needed: a Gemini API key and BigQuery access. Neither requires a
credit card.

---

## 1. Gemini API key (2 minutes)

1. Go to <https://aistudio.google.com/apikey>
2. *Create API key* → copy it
3. `cp .env.example .env` and set `GOOGLE_API_KEY`

**Free-tier limits are tight** (~10 requests/minute, 250/day, 250k tokens/minute). The agent is
designed around this: 3 or fewer LLM calls per turn, and a record/replay cache so repeated
development runs cost nothing.

## 2. BigQuery access

Public-dataset *storage* is free, but queries bill to your own project, so you need a
project. The BigQuery Sandbox gives 1 TiB/month of free compute with no billing account.

1. Create a project at <https://console.cloud.google.com/projectcreate>
2. Enable the BigQuery API:
   <https://console.cloud.google.com/apis/library/bigquery.googleapis.com>
3. Set `GCP_PROJECT_ID` in `.env`

Then pick one authentication path:

### Path A: gcloud ADC (recommended)

No key file exists, so there is nothing secret to leak or accidentally commit.

```powershell
winget install Google.CloudSDK
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Leave `GOOGLE_APPLICATION_CREDENTIALS` unset. Restart the shell after installing so `gcloud`
lands on `PATH`.

### Path B: service-account JSON (no SDK install)

Faster if the SDK install is awkward, but produces a secret file.

1. <https://console.cloud.google.com/iam-admin/serviceaccounts> → *Create service account*
2. Grant BigQuery User (`roles/bigquery.user`), needed to run query jobs
3. *Keys* → *Add key* → *JSON* → download
4. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env` to that path

The key is a credential. `.gitignore` already excludes `*-service-account*.json`,
`*credentials*.json` and `gcp-key*.json`, keep it outside the repo anyway.

## 3. Install and verify

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m analyst_assistant doctor
```

`doctor` checks credentials, lists the Gemini models this key can actually reach, and
dry-runs one BigQuery query. Run it before anything else, credentials and model availability
are the two things most likely to derail a build, and both fail in confusing ways later.

### Known trap

`to_dataframe()` raises without `db-dtypes` installed, because `thelook_ecommerce` has DATE,
TIMESTAMP and NUMERIC columns. It is pinned in `requirements.txt` for this reason. If you run
the assignment's `b1_client.py` standalone, install `db-dtypes` and `pyarrow` first.

## 4. No Google Cloud project? Replay mode

```powershell
$env:BQ_MODE = "replay"
python -m analyst_assistant chat
```

Replay answers from recorded query results in `fixtures/`, so **step 2 is not needed**.

Be clear on what this does and does not remove. It removes BigQuery, not Gemini: triage,
SQL generation and the written answer are all model calls, so **a Gemini API key is still
required** and the agent will refuse to start without one. It is also only as complete as
the recordings: the lookup is keyed on the generated SQL, so a question whose SQL does not
match a fixture reports a replay miss rather than an answer.

What it is good for: the safety, confirmation, resilience and observability paths, none of
which touch the warehouse. The saved-report and delete flows work fully in replay mode.
For a reviewer who wants to see a complete analysis session without running anything, the
transcript in the README is the better artifact.
