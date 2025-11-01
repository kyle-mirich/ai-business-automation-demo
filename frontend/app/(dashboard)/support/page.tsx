'use client';

import { useState } from 'react';
import { Send, AlertCircle, CheckCircle2, Clock, DollarSign, TrendingUp, Sparkles } from 'lucide-react';
import { api } from '@/lib/api';
import type { SupportTicketResponse } from '@/lib/types';
import { Button } from '@/components/ui/Button';
import { Input, Textarea } from '@/components/ui/Input';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { getPriorityColor, formatCurrency } from '@/lib/utils';

// Demo tickets from data/tickets.json
const DEMO_TICKETS = [
  {
    id: 1,
    subject: "Delayed Order #87345",
    message: "Hi team, I'm still waiting for order #87345. It was supposed to arrive two weeks ago and the tracking has not updated. Can someone help me figure out where it is?",
    customer: {
      name: "John Smith",
      email: "john.smith@example.com",
      tier: "premium"
    }
  },
  {
    id: 2,
    subject: "Damaged monitor arrived",
    message: "I received the 27\" 4K monitor today but the screen is cracked right out of the box. I need a refund or a replacement immediately. This is really disappointing.",
    customer: {
      name: "Alicia Gomez",
      email: "alicia.gomez@example.com",
      tier: "standard"
    }
  },
  {
    id: 3,
    subject: "Dashboard down for enterprise customers",
    message: "We cannot access the admin analytics dashboard this morning. It just spins and times out. We use this daily and need it back ASAP.",
    customer: {
      name: "Renee Williams",
      email: "renee.williams@brightcore.io",
      tier: "business"
    }
  }
];

export default function SupportTriagePage() {
  const [selectedTicketIndex, setSelectedTicketIndex] = useState(0);
  const [formData, setFormData] = useState({
    ticket_id: `T-${DEMO_TICKETS[0].id}`,
    subject: DEMO_TICKETS[0].subject,
    description: DEMO_TICKETS[0].message,
    customer_email: DEMO_TICKETS[0].customer.email,
    customer_name: DEMO_TICKETS[0].customer.name,
  });
  const [result, setResult] = useState<SupportTicketResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.triageTicket(formData);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  const loadTicket = (index: number) => {
    const ticket = DEMO_TICKETS[index];
    setSelectedTicketIndex(index);
    setFormData({
      ticket_id: `T-${ticket.id}`,
      subject: ticket.subject,
      description: ticket.message,
      customer_email: ticket.customer.email,
      customer_name: ticket.customer.name,
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-purple-700 rounded-xl p-8 text-white shadow-lg">
        <div className="flex items-center space-x-4 mb-4">
          <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
            <Sparkles className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Support Ticket Triage</h1>
            <p className="text-purple-100 mt-1">
              Powered by LangChain Multi-Agent System with Gemini AI
            </p>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mt-6">
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="text-2xl font-bold">Auto-Classify</div>
            <div className="text-sm text-purple-100">Intelligent Routing</div>
          </div>
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="text-2xl font-bold">Prioritize</div>
            <div className="text-sm text-purple-100">Urgency Detection</div>
          </div>
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="text-2xl font-bold">AI Response</div>
            <div className="text-sm text-purple-100">Automated Drafts</div>
          </div>
        </div>
      </div>

      {/* Demo Ticket Selector */}
      <Card className="bg-gradient-to-r from-gray-50 to-gray-100">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-900">Demo Tickets</h3>
            <Badge variant="info">Live API Calls</Badge>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {DEMO_TICKETS.map((ticket, index) => (
              <button
                key={ticket.id}
                onClick={() => loadTicket(index)}
                className={`p-4 rounded-lg border-2 text-left transition-all ${
                  selectedTicketIndex === index
                    ? 'border-purple-500 bg-white shadow-md'
                    : 'border-gray-200 bg-white/50 hover:border-purple-300'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <Badge variant={index === selectedTicketIndex ? 'info' : 'default'} className="text-xs">
                    T-{ticket.id}
                  </Badge>
                  <span className="text-xs text-gray-500 capitalize">{ticket.customer.tier}</span>
                </div>
                <h4 className="font-medium text-sm text-gray-900 mb-1 line-clamp-1">{ticket.subject}</h4>
                <p className="text-xs text-gray-600 line-clamp-2">{ticket.message}</p>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle>Current Ticket</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Ticket ID
                </label>
                <Input
                  value={formData.ticket_id}
                  onChange={(e) => setFormData({ ...formData, ticket_id: e.target.value })}
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Customer Name
                </label>
                <Input
                  value={formData.customer_name}
                  onChange={(e) => setFormData({ ...formData, customer_name: e.target.value })}
                  placeholder="Optional"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Customer Email
                </label>
                <Input
                  type="email"
                  value={formData.customer_email}
                  onChange={(e) => setFormData({ ...formData, customer_email: e.target.value })}
                  required
                  placeholder="customer@example.com"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Subject
                </label>
                <Input
                  value={formData.subject}
                  onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                  required
                  placeholder="Brief description of the issue"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Description
                </label>
                <Textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  required
                  rows={6}
                  placeholder="Detailed description of the issue..."
                />
              </div>

              {error && (
                <div className="bg-red-50 border border-red-200 rounded-md p-3 flex items-start space-x-2">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              )}

              <div className="flex gap-2">
                <Button type="submit" disabled={isLoading} className="flex-1">
                  {isLoading ? (
                    <>
                      <LoadingSpinner size="sm" className="mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4 mr-2" />
                      Analyze Ticket
                    </>
                  )}
                </Button>
                {result && (
                  <Button type="button" variant="outline" onClick={handleReset}>
                    Clear
                  </Button>
                )}
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Results */}
        <div className="space-y-6">
          {result ? (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-2 gap-4">
                <Card className="p-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm text-gray-500">Category</p>
                      <p className="text-lg font-semibold text-gray-900 mt-1">
                        {result.category.replace(/_/g, ' ')}
                      </p>
                    </div>
                    <Badge variant="info">
                      {(result.category_confidence * 100).toFixed(0)}%
                    </Badge>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm text-gray-500">Priority</p>
                      <p className="text-lg font-semibold text-gray-900 mt-1">
                        {result.priority}
                      </p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${getPriorityColor(result.priority)}`}>
                      Score: {(result.priority_score * 100).toFixed(0)}
                    </div>
                  </div>
                </Card>

                <Card className="p-4">
                  <div>
                    <p className="text-sm text-gray-500">Department</p>
                    <p className="text-lg font-semibold text-gray-900 mt-1">
                      {result.department}
                    </p>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="flex items-start space-x-2">
                    <DollarSign className="w-5 h-5 text-gray-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm text-gray-500">API Cost</p>
                      <p className="text-lg font-semibold text-gray-900 mt-1">
                        {formatCurrency(result.cost_usd)}
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              {/* AI Generated Response */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                    <span>AI-Generated Response</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-gray-900 whitespace-pre-wrap">{result.response}</p>
                  </div>
                </CardContent>
              </Card>

              {/* Priority Reasoning */}
              {result.priority_reasoning && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <TrendingUp className="w-5 h-5 text-blue-600" />
                      <span>Priority Analysis</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-700 text-sm">{result.priority_reasoning}</p>
                  </CardContent>
                </Card>
              )}

              {/* Processing Steps */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Clock className="w-5 h-5 text-gray-600" />
                    <span>Processing Steps</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {result.steps.map((step, index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <div className="w-6 h-6 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                          <span className="text-xs font-medium text-primary-700">{index + 1}</span>
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900">{step.agent}</p>
                          <p className="text-sm text-gray-600">{step.summary}</p>
                          {step.token_usage && (
                            <p className="text-xs text-gray-500 mt-1">
                              Tokens: {step.token_usage.total_tokens}
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle>Usage Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <dt className="text-gray-500">Total Tokens</dt>
                      <dd className="text-lg font-semibold text-gray-900 mt-1">
                        {result.tokens_used.toLocaleString()}
                      </dd>
                    </div>
                    <div>
                      <dt className="text-gray-500">Estimated Cost</dt>
                      <dd className="text-lg font-semibold text-gray-900 mt-1">
                        {formatCurrency(result.cost_usd)}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card className="p-12">
              <div className="text-center text-gray-500">
                <AlertCircle className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>Submit a ticket to see AI-powered analysis</p>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
