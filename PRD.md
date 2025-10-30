# Build AI Business Automation Demo - Instructions for LLM

You are an expert Python developer building a Streamlit application that demonstrates AI agents automating real business workflows. This is a portfolio demo designed to impress recruiters at AI companies.

---

## Project Overview

**Goal:** Build a single Streamlit application with 3 interactive demos that show AI agents working in real-time. Users should see the actual prompts sent to the LLM, the responses streaming back, and the final results - making AI transparent and impressive.

**Key Requirements:**
1. Use **pre-loaded data** (no file uploads) to ensure zero errors during live demos
2. Make **real AI API calls** (not mocked) - actual Gemini API, actual Prophet forecasting
3. Show **"behind the scenes"** - display prompts, LLM responses, agent decisions in real-time
4. Build everything in **pure Python** - single Streamlit app, no separate APIs or microservices
5. Design for **interview demos** - must work flawlessly in 2-5 minutes

**Tech Stack:**
- **Framework:** Streamlit (for UI)
- **AI/Agents:** LangChain, LangGraph (for multi-agent orchestration)
- **LLM:** Google Generative AI (Gemini API)
- **Forecasting:** Prophet (Facebook's time series forecasting)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly (interactive charts)

---

## Project Structure

Create the following structure:

```
ai-demo-streamlit/
├── app.py                          # Homepage with navigation
├── pages/
│   ├── 1_📊_Financial_Report.py   # Demo 1: Financial analysis
│   ├── 2_🎫_Support_Triage.py     # Demo 2: Multi-agent support
│   └── 3_📦_Inventory_Optimizer.py # Demo 3: Inventory optimization
├── agents/
│   ├── financial_agent.py          # Agent for financial analysis
│   ├── support_agents.py           # Multi-agent system with LangGraph
│   └── inventory_agent.py          # Agent for inventory optimization
├── data/
│   ├── sales_2025_q3.csv          # Pre-generated Q3 sales data
│   ├── tickets.json                # Pre-generated support tickets
│   └── inventory.csv               # Pre-generated inventory data
├── utils/
│   ├── data_loader.py             # Helper functions for loading data
│   └── cost_calculator.py         # Calculate API costs
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (GOOGLE_API_KEY)
└── README.md                      # Project documentation
```

---

## Demo 1: Financial Report Generator (Priority: HIGH - Build This First)

### What It Does
Analyzes pre-loaded Q3 2025 sales data using Gemini API to generate insights, then uses Prophet to forecast Q4 revenue. Shows the entire process in real-time with visualizations.

### User Flow
1. User navigates to "📊 Financial Report" page in sidebar
2. Sees a button: **"Process sales_2025_q3.csv"**
3. Clicks button
4. Progress bar appears with status updates
5. "Behind the Scenes" expander auto-opens showing:
   - Step 1: Loading data (shows row count, date range, total revenue)
   - Step 2: Analyzing with Gemini (shows actual prompt sent + streaming LLM response)
   - Step 3: Forecasting with Prophet (shows training status + predicted Q4 revenue)
   - Step 4: Generating visualizations (shows completion)
6. Results display in main area:
   - AI-generated insights (revenue trends, top products, recommendations)
   - 5 interactive Plotly charts (revenue trend, top products, category breakdown, customer segments, Q4 forecast)
   - Metrics summary (Q3 revenue, Q4 forecast, growth rate, confidence)
   - API cost and execution time
7. User can run analysis again or switch demos

### Technical Requirements

**Data File: `data/sales_2025_q3.csv`**
- Generate 247 rows of realistic e-commerce sales data
- Columns: date, product, quantity, revenue, cost, category, customer_segment
- Date range: July 1 - September 30, 2025
- Include ~10 products (Wireless Headphones, USB-C Cable, Laptop Stand, etc.)
- 3 categories: Electronics, Accessories, Office
- 3 customer segments: Consumer, Business, Education
- Total revenue should be around $124,000
- **Important:** Include intentional growth trend from July → September for interesting insights

**Agent Implementation: `agents/financial_agent.py`**

Create a `FinancialAgent` class that:

1. **Loads data** from CSV using Pandas
   - Parse dates correctly
   - Calculate summary statistics (total revenue, row count, date range)
   - Aggregate data by product, category, segment, month

2. **Analyzes with Gemini API**
   - Create a detailed prompt asking for:
     * Revenue trends (July → September growth)
     * Top 5 products by revenue with dollar amounts
     * Category performance comparison
     * Customer segment analysis
     * Areas of concern or declining products
     * Growth opportunities for Q4
   - Use `google.generativeai` package to call Gemini
   - Stream the response (use `stream=True`) so it displays in real-time
   - Set temperature to 0.3 for consistent analysis
   - Track token usage for cost calculation

3. **Forecasts with Prophet**
   - Convert daily revenue data to Prophet format (columns: 'ds' for date, 'y' for revenue)
   - Initialize Prophet model with yearly seasonality
   - Fit model on Q3 data
   - Predict next 90 days (Q4)
   - Calculate: predicted Q4 total, growth rate vs Q3, confidence intervals
   - Return forecast dataframe for visualization

4. **Generates Plotly charts**
   - Revenue trend line chart (Q3 daily revenue)
   - Top 10 products horizontal bar chart
   - Category performance pie chart
   - Customer segment bar chart
   - Q4 forecast chart showing:
     * Q3 actual (solid blue line)
     * Q4 forecast (dashed green line)
     * 95% confidence interval (shaded area)

5. **Calculates API cost**
   - Estimate tokens used (word count * 1.5)
   - Use Gemini pricing: ~$0.001 per 1K tokens
   - Return cost in dollars

**Streamlit Page: `pages/1_📊_Financial_Report.py`**

Build a Streamlit page that:
- Shows title and description
- Has a primary button to trigger analysis
- Uses `st.progress()` for progress bar
- Uses `st.status()` with `expanded=True` for each step in "Behind the Scenes"
- Shows the actual prompt in a code block (`st.code()`)
- Streams Gemini response into a placeholder that updates in real-time
- Displays all 5 Plotly charts using `st.plotly_chart()`
- Shows metrics using `st.metric()` (Q3 revenue, Q4 forecast, cost, time)
- Uses `st.session_state` to store results so they persist
- Includes a "Run Again" button after completion

**Key Features to Highlight:**
- The prompt should be visible and well-formatted - this shows prompt engineering skills
- Stream the LLM response word-by-word or chunk-by-chunk for dramatic effect
- Prophet forecast should show actual training happening (not instant)
- Charts should be interactive (Plotly hover effects)
- Metrics should be prominently displayed

---

## Demo 2: Customer Support Ticket Triage (Priority: MEDIUM)

### What It Does
Uses a multi-agent system built with LangGraph to process 5 support tickets through 4 coordinated agents: Classifier → Prioritizer → Router → Response Generator. Shows how agents pass information between each other.

### User Flow
1. User navigates to "🎫 Support Triage" page
2. Sees button: **"Process 5 Pending Tickets"**
3. Clicks button
4. 5 ticket cards appear showing customer names and message previews
5. Each ticket is processed sequentially through the agent pipeline:
   - Classifier Agent: Categorizes ticket (SHIPPING_INQUIRY, REFUND_REQUEST, etc.)
   - Priority Agent: Assigns urgency (CRITICAL, HIGH, MEDIUM, LOW)
   - Router Agent: Routes to department (Logistics, Refunds, Sales, etc.)
   - Response Agent: Drafts customer reply
6. "Behind the Scenes" shows agent coordination for each ticket
7. Each ticket card updates with final results: category, priority, department, response
8. Summary shows category breakdown and priority distribution

### Technical Requirements

**Data File: `data/tickets.json`**

Create 5 realistic support tickets in JSON format:
```json
[
  {
    "id": 1,
    "message": "Where is my order #12345? It's been 2 weeks!",
    "customer": {"name": "John Smith", "email": "...", "tier": "premium"},
    "timestamp": "2025-10-30T10:30:00Z"
  },
  // ... 4 more tickets
]
```

Include variety:
- Shipping inquiry (frustrated customer, 2 weeks delay)
- Refund request (damaged product, angry tone)
- Address change (simple, neutral tone)
- Bulk sales inquiry (business opportunity)
- Technical issue (website down, critical urgency)

Mix customer tiers: premium, standard, business

**Agent Implementation: `agents/support_agents.py`**

Create `SupportAgentOrchestrator` class using **LangGraph** that:

1. **Defines agent state** (TypedDict with: ticket, category, priority, department, response)

2. **Implements 4 agents as separate methods:**

   **Classifier Agent:**
   - Takes ticket message
   - Prompts Gemini to classify into one category: SHIPPING_INQUIRY, REFUND_REQUEST, ADDRESS_CHANGE, BULK_SALES, TECHNICAL_ISSUE, PRODUCT_QUESTION, ACCOUNT_ISSUE
   - Returns category + confidence score (0-100)
   
   **Prioritizer Agent:**
   - Takes ticket + category
   - Prompts Gemini to assign priority: CRITICAL, HIGH, MEDIUM, LOW
   - Considers: issue urgency, customer sentiment, customer tier, business impact
   - Returns priority + score + reasoning
   
   **Router Agent:**
   - Takes category
   - Uses rule-based mapping to assign department
   - No LLM needed - just a dictionary lookup
   - Returns department name
   
   **Response Agent:**
   - Takes ticket + category + priority + department
   - Prompts Gemini to draft professional customer reply
   - Response should match urgency level (critical = immediate action language)
   - Keep 3-4 sentences, professional and empathetic
   - Returns draft response

3. **Builds LangGraph workflow:**
   - Create StateGraph
   - Add 4 nodes (one per agent)
   - Connect them: classifier → prioritizer → router → responder → END
   - Compile the graph

4. **Provides helper methods:**
   - `process_ticket()` - runs a single ticket through the workflow
   - `classify_ticket()` - exposes just classification
   - `calculate_cost()` - tracks API usage

**Streamlit Page: `pages/2_🎫_Support_Triage.py`**

Build a page that:
- Shows 5 ticket cards in columns (use `st.columns(5)`)
- Each card shows customer name and message preview
- Uses `st.status()` within each card to show processing steps
- "Behind the Scenes" expander shows agent coordination for each ticket
- For each agent, show:
  * Agent name
  * What it's analyzing
  * Its output/decision
- Update ticket cards progressively as agents complete
- Final ticket cards show: category, priority, department, expandable response
- Show summary metrics: total time, API cost, tickets processed
- Show breakdown: categories (how many of each), priorities (distribution)

**LangGraph Usage:**
- This demo showcases multi-agent coordination
- Make sure to show that data flows: Classifier output → Prioritizer input → etc.
- Each agent should have visible output in the UI
- This proves you understand agent orchestration

---

## Demo 3: Inventory Optimizer (Priority: LOW - If Time Permits)

### What It Does
Analyzes inventory levels, forecasts demand for each product using Prophet, and uses Gemini to generate prioritized reorder recommendations.

### User Flow
1. User navigates to "📦 Inventory Optimizer" page
2. Sees button: **"Analyze Current Inventory"**
3. Current inventory table displays (50 SKUs)
4. Agent analyzes:
   - Items below reorder point (urgent)
   - Slow-moving stock (overstock issues)
   - Fast-moving items (high demand)
5. Prophet forecasts 30-day demand for each product
6. Gemini generates prioritized reorder list with reasoning
7. Results show:
   - Inventory alerts (low stock items)
   - Demand forecasts visualization
   - Reorder recommendations table (priority, quantity, reasoning, cost)

### Technical Requirements

**Data File: `data/inventory.csv`**

Generate 50 inventory items:
- Columns: sku, product_name, current_stock, reorder_point, lead_time_days, cost_per_unit, last_30_days_sales, category
- Include intentional issues:
  * 5 items critically low (current_stock < reorder_point)
  * 3 items overstocked (high stock, low sales)
  * 10 fast-moving items (high sales)
- Realistic products: electronics, accessories, office supplies

**Agent Implementation: `agents/inventory_agent.py`**

Create `InventoryAgent` class that:

1. **Analyzes inventory levels**
   - Load CSV
   - Identify low stock items (current < reorder_point)
   - Identify slow movers (sales < 10 units, high stock)
   - Identify fast movers (sales > 50 units)

2. **Forecasts demand with Prophet**
   - For each product, create simple historical data (30 days)
   - Fit Prophet model
   - Predict next 30 days
   - Sum predicted demand

3. **Generates reorder recommendations with Gemini**
   - Create detailed prompt with:
     * Items needing reorder
     * Current stock vs reorder point
     * Forecasted demand
     * Lead times
     * Costs
   - Ask Gemini to recommend:
     * Order quantity (considering forecast + safety stock + lead time)
     * Priority (HIGH/MEDIUM/LOW)
     * Reasoning
     * Estimated cost
   - Parse JSON response from Gemini

4. **Creates visualizations**
   - Inventory levels bar chart (color-coded: red = low, yellow = ok, green = good)
   - Demand forecast line charts for top items
   - Reorder priority breakdown

**Streamlit Page: `pages/3_📦_Inventory_Optimizer.py`**

Build a page showing:
- Current inventory table with highlighting
- "Behind the Scenes" with analysis steps
- Prophet forecasting progress
- Gemini recommendation generation
- Final reorder list as interactive table
- Charts and metrics

---

## Homepage: `app.py`

Build a Streamlit homepage that:

1. **Header section:**
   - Title: "🤖 AI Business Automation Demo"
   - Subtitle: "Watch AI agents automate real business workflows in real-time"
   - Description: "See actual prompts, LLM responses, and results - no setup required"

2. **Demo cards:**
   - Use `st.columns(3)` to create 3 cards
   - Each card has:
     * Icon + Title
     * Description of what it does
     * Tech stack badges (e.g., "LangChain, Prophet, Plotly")
     * Link to demo page using `st.page_link()`

3. **Footer section:**
   - "Built with:" list of technologies
   - Feature highlights (Real AI calls, Pre-loaded data, Single deployment, etc.)
   - GitHub link

**Design:**
- Use Streamlit's native styling
- Keep it clean and professional
- Blue color scheme (#2563eb primary)
- No custom CSS needed

---

## Configuration Files

### `requirements.txt`
```
streamlit>=1.31.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.18.0
prophet>=1.1.5
google-generativeai>=0.3.2
langgraph>=0.0.26
langchain>=0.1.0
langchain-google-genai>=0.0.9
python-dotenv>=1.0.0
```

### `.env`
```
GOOGLE_API_KEY=your_api_key_here
```

### `.streamlit/config.toml`
```toml
[theme]
primaryColor="#2563eb"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f1f5f9"
textColor="#1e293b"
font="sans serif"

[server]
headless = true
port = 8501
```

---

## Key Technical Specifications

### Gemini API Usage
- Use `google.generativeai` package
- Model: `gemini-2.5-flash` for financial analysis (more capable)
- Model: `gemini-1.5-flash` for classification/support (faster, cheaper)
- Set appropriate temperature:
  * 0.1-0.2 for classification (consistent)
  * 0.3 for analysis (balanced)
  * 0.5 for creative responses (if needed)
- Always stream responses when showing to user (`stream=True`)
- Track tokens for cost calculation

### Prophet Usage
- Use `Prophet` from `prophet` package
- Always set seasonality appropriately:
  * yearly_seasonality=True for long trends
  * weekly_seasonality=False for daily data
  * daily_seasonality=False for aggregated data
- Use `.fit()` for training
- Use `.make_future_dataframe(periods=N)` for forecasting
- Use `.predict()` to get forecast
- Extract 'yhat' (prediction), 'yhat_lower', 'yhat_upper' (confidence)

### LangGraph Usage (Support Demo)
- Import: `from langgraph.graph import StateGraph, END`
- Define state with TypedDict
- Create graph: `StateGraph(StateType)`
- Add nodes: `graph.add_node("name", function)`
- Add edges: `graph.add_edge("from", "to")`
- Set entry point: `graph.set_entry_point("name")`
- Compile: `graph.compile()`
- Invoke: `graph.invoke(initial_state)`

### Streamlit Best Practices
- Use `st.session_state` to persist results between reruns
- Use `st.progress()` for progress bars
- Use `st.status()` for expandable step-by-step updates
- Use `st.expander()` for "Behind the Scenes" sections
- Use `st.columns()` for side-by-side layouts
- Use `st.metric()` for displaying metrics with labels
- Use `st.plotly_chart()` for interactive charts
- Use `st.code()` for displaying code/prompts
- Always include `use_container_width=True` for responsive charts

### Cost Tracking
- Track all API calls
- Estimate tokens: word_count * 1.5
- Gemini Pro pricing: ~$0.001 per 1K tokens
- Gemini Flash pricing: ~$0.0005 per 1K tokens
- Display cost to 3-4 decimal places (e.g., $0.043)

### Error Handling
- Wrap agent calls in try-except blocks
- Use `st.error()` to display errors
- Gracefully handle API failures
- Validate data before processing
- Log errors for debugging

---

## Data Generation Guidelines

### Sales Data (`sales_2025_q3.csv`)
- 247 rows (about 2-3 transactions per day for 92 days)
- Realistic product names and prices
- Include growth trend: July < August < September
- Some products should be popular, others less so
- Mix of customer segments
- Revenue range: $5-2000 per transaction
- Ensure data is clean (no nulls, valid dates, positive numbers)

### Tickets Data (`tickets.json`)
- 5 diverse tickets
- Mix of urgent and non-urgent
- Mix of customer tiers
- Realistic language (some frustrated, some neutral)
- Include specific details (order numbers, time frames)
- Vary message lengths (50-150 characters)

### Inventory Data (`inventory.csv`)
- 50 SKUs
- Realistic products matching sales data products
- Intentionally include:
  * 5+ items below reorder point (stock out risk)
  * 3+ items with low sales but high stock (overstock)
  * 10+ items with high sales (fast movers)
- Varied lead times (3-21 days)
- Varied costs ($5-200)
- Sales data for last 30 days (0-200 units)

---

## UI/UX Requirements

### General Design
- Clean, professional appearance
- Blue color scheme (primary: #2563eb)
- Clear hierarchies with headers
- Generous whitespace
- Mobile-responsive (Streamlit handles this)

### "Behind the Scenes" Panel
- Always use expander for collapsible view
- Show 4 clear steps for each demo
- Use status indicators: ✅ (complete), ⏳ (in progress), ⏸ (pending)
- Display actual prompts in code blocks
- Stream LLM responses visibly
- Show cost and time at the bottom

### Results Display
- Lead with insights (text)
- Follow with visualizations (charts)
- End with metrics (numbers)
- Use cards or containers for grouping
- Include download options where relevant

### Interactive Elements
- Primary buttons should be prominent (blue, full-width)
- "Run Again" buttons should be secondary (gray)
- Use expanders for optional details
- Include helpful descriptions everywhere

---

## Success Criteria

Your implementation is successful if:

✅ All 3 demos work without errors  
✅ Real Gemini API calls are made (responses are unique each run)  
✅ Prophet forecasting produces reasonable predictions  
✅ "Behind the Scenes" makes AI transparent (shows prompts and responses)  
✅ UI is clean and professional  
✅ Data is pre-loaded (no file uploads)  
✅ App runs locally with `streamlit run app.py`  
✅ No errors in console  
✅ Response time is reasonable (<15 seconds per demo)  
✅ Cost tracking is accurate  
✅ Can be deployed to Streamlit Cloud easily  

---

## Development Order (Recommended)

**Day 1: Foundation + Financial Demo**
1. Set up project structure
2. Install dependencies
3. Create `.env` with API key
4. Generate `sales_2025_q3.csv`
5. Build `FinancialAgent` class
6. Build Financial Report page
7. Test thoroughly

**Day 2: Support Demo**
8. Generate `tickets.json`
9. Build `SupportAgentOrchestrator` with LangGraph
10. Build Support Triage page
11. Test multi-agent flow

**Day 3: Inventory Demo + Polish**
12. Generate `inventory.csv`
13. Build `InventoryAgent`
14. Build Inventory Optimizer page
15. Polish homepage
16. Add README
17. Test all demos end-to-end

---

## Testing Checklist

Before considering the project complete, verify:

- [ ] Financial demo: Loads data, analyzes with Gemini, forecasts with Prophet, shows charts
- [ ] Support demo: Processes all 5 tickets through all 4 agents correctly
- [ ] Inventory demo: Identifies issues, forecasts demand, generates recommendations
- [ ] All prompts are visible in "Behind the Scenes"
- [ ] All LLM responses stream visibly (not instant)
- [ ] All charts render correctly and are interactive
- [ ] Cost calculations are shown and reasonable
- [ ] No errors in terminal when running
- [ ] Can run demo multiple times and get different results
- [ ] Session state preserves results
- [ ] App is responsive on different screen sizes

---

## Deployment Instructions

**To Streamlit Cloud:**
1. Push code to GitHub repository
2. Go to streamlit.io/cloud
3. Connect GitHub account
4. Select repository
5. Set main file: `app.py`
6. Add secret: `GOOGLE_API_KEY` in Streamlit Cloud settings
7. Deploy

**To Run Locally:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Important Notes

1. **Pre-loaded data is critical** - This ensures demos work perfectly every time without user error
2. **Real AI calls are essential** - Do not mock responses; this demo proves AI capabilities
3. **Transparency is the killer feature** - Showing prompts and responses makes this special
4. **Streamlit simplicity** - Use native Streamlit components, don't over-engineer
5. **Interview focus** - Design every feature thinking "will this impress a recruiter?"

---

## README.md Template

Include in the README:
- Project overview
- Features (3 demos description)
- Tech stack
- Installation instructions
- Usage guide
- Demo screenshots (optional)
- Architecture overview
- API key setup
- Deployment guide

---

## Final Checklist

✅ Project structure created  
✅ All dependencies installed  
✅ Data files generated  
✅ All 3 agents implemented  
✅ All 3 Streamlit pages built  
✅ Homepage with navigation  
✅ "Behind the Scenes" working  
✅ Real-time streaming working  
✅ Charts displaying correctly  
✅ Cost tracking accurate  
✅ Error handling implemented  
✅ Tested locally  
✅ README written  
✅ Ready to deploy  

---

**You are building a portfolio demo that will differentiate Kyle in AI operations/solutions interviews. Focus on making AI transparent, reliable, and impressive. Show the prompts, show the thinking, show the results. This is not just a technical demo - it's a conversation starter about how AI transforms business operations.**

**Good luck! Build something amazing! 🚀**