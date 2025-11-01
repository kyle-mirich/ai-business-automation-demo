# AI Business Automation - Next.js Frontend

🚀 **Production-ready Next.js application showcasing enterprise AI automation capabilities**

## Overview

This is a modern, full-stack demonstration platform built to showcase professional-grade AI integrations for business automation. Perfect for showing hiring managers and technical recruiters your expertise in:

- Modern React/Next.js development
- TypeScript proficiency
- API integration
- Real-time streaming
- Professional UI/UX design
- Production-ready code quality

## Live Demo Features

### 1. RAG Chatbot (Retrieval-Augmented Generation)
- Query 12 AI research papers with intelligent semantic search
- Real-time streaming responses using Server-Sent Events
- Source citations with page numbers and file paths
- Conversation history management
- Copy-to-clipboard functionality
- Token usage and cost tracking

### 2. Support Ticket Triage
- Automated ticket classification (8 categories)
- AI-powered priority scoring (Critical/High/Medium/Low)
- Department routing recommendations
- AI-generated customer responses
- Processing steps visualization
- Real-time cost calculation

### 3. Inventory Optimizer
- Upload CSV or analyze default dataset
- AI-powered demand forecasting (30/60/90 days)
- Reorder recommendations with urgency levels
- Low stock and overstock detection
- Interactive data tables with sorting/filtering
- Visual dashboards with charts
- Cost impact analysis

## Tech Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Charts**: Recharts
- **Animations**: Framer Motion

### Backend Integration
- FastAPI (Python)
- Google Gemini 2.5 Flash
- LangChain
- ChromaDB Vector Store

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend API running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/                        # Next.js App Router
│   ├── layout.tsx             # Root layout
│   ├── page.tsx               # Landing page
│   ├── globals.css            # Global styles
│   └── (dashboard)/           # Protected routes
│       ├── layout.tsx         # Dashboard layout
│       ├── rag/page.tsx       # RAG Chatbot
│       ├── support/page.tsx   # Support Triage
│       └── inventory/page.tsx # Inventory Optimizer
│
├── components/                 # React components
│   ├── ui/                    # Reusable UI elements
│   └── features/              # Feature-specific components
│
├── lib/                        # Utilities
│   ├── api.ts                 # API client
│   ├── types.ts               # TypeScript types
│   └── utils.ts               # Helper functions
│
├── public/                     # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.ts
```

## Key Features for Recruiters

### 1. TypeScript Excellence
- Fully typed API responses and requests
- Interface definitions for all data structures
- Type-safe API client with generics
- Proper error handling with custom types

### 2. Modern React Patterns
- Server and Client Components
- Streaming with AsyncGenerators
- Optimistic UI updates
- Proper state management
- Error boundaries and suspense

### 3. Performance Optimization
- Code splitting
- Dynamic imports
- Image optimization
- CSS-in-JS with Tailwind
- Lazy loading

### 4. Professional UI/UX
- Responsive design (mobile-first)
- Accessibility (ARIA labels, keyboard nav)
- Loading skeletons
- Error states with retry
- Smooth animations
- Toast notifications

### 5. Production-Ready Code
- ESLint configuration
- Consistent code style
- Component composition
- DRY principles
- Separation of concerns
- Comprehensive error handling

## API Integration

### Endpoints Used
- `POST /api/rag/query` - RAG chatbot queries
- `POST /api/rag/query/stream` - Streaming responses
- `POST /api/support/triage` - Ticket classification
- `POST /api/inventory/analyze` - Inventory analysis
- `POST /api/inventory/analyze/upload` - CSV upload

### Error Handling
```typescript
try {
  const response = await api.queryRAG({ query, top_k: 3 });
  // Handle success
} catch (error) {
  // User-friendly error message
  console.error('Failed to query:', error);
}
```

### Streaming Implementation
```typescript
const stream = api.queryRAGStream(request);
for await (const chunk of stream) {
  // Handle each chunk
  updateUI(chunk);
}
```

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Deployment

### Vercel (Recommended)
```bash
vercel --prod
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## What Makes This Impressive

1. ✅ **Production-Ready**: Not a prototype - deployment-ready code
2. ✅ **Best Practices**: Follows Next.js and React conventions
3. ✅ **Type Safety**: 100% TypeScript coverage
4. ✅ **Modern Stack**: Latest Next.js features (App Router, Server Components)
5. ✅ **Performance**: Optimized for speed and UX
6. ✅ **Scalable**: Easy to extend with new features
7. ✅ **Professional Design**: Clean, modern UI
8. ✅ **Real-time Features**: Streaming, live updates
9. ✅ **Accessibility**: WCAG compliant
10. ✅ **Documentation**: Comprehensive README and code comments

## Demo Script for Recruiters

### 1. Landing Page
"This is the entry point showcasing all three AI features with modern design and animations."

### 2. RAG Chatbot
"Real-time AI responses with source citations. Notice the streaming effect and how sources are displayed with file paths and page numbers. This demonstrates WebSocket/SSE integration."

### 3. Support Triage
"Submit a customer ticket and watch AI classify, prioritize, route, and generate a response in real-time. Notice the step-by-step visualization and cost tracking."

### 4. Inventory Optimizer
"Upload inventory data or use the default dataset. AI analyzes stock levels, forecasts demand, and provides actionable recommendations with visual dashboards."

## Technical Highlights

- **Real-time Streaming**: Server-Sent Events for live AI responses
- **Type Safety**: Full TypeScript coverage with strict mode
- **Error Handling**: Comprehensive try/catch with user-friendly messages
- **Loading States**: Skeletons and spinners everywhere
- **Responsive**: Works on mobile, tablet, and desktop
- **Accessible**: Keyboard navigation, ARIA labels, screen reader support
- **Performance**: Code splitting, lazy loading, optimized renders
- **Clean Code**: DRY, SOLID principles, proper separation

## License

MIT

## Author

Kyle Mirich
- GitHub: [@kyle-mirich](https://github.com/kyle-mirich)
- Portfolio: [Your Portfolio URL]

---

**Built to impress technical recruiters and hiring managers** 🎯
