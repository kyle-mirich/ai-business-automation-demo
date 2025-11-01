'use client';

import { useState, useEffect } from 'react';
import { DollarSign, TrendingUp, Send, Bot, User, Code, Database, BarChart3, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import dynamic from 'next/dynamic';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github-dark.css';
import { api } from '@/lib/api';
import type { FinancialStatsResponse, FinancialChatResponse, FinancialReportResponse, ChatMessage, ToolUsage } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { LoadingSpinner, LoadingDots } from '@/components/ui/LoadingSpinner';
import { formatCurrency } from '@/lib/utils';
import { cn } from '@/lib/utils';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ChatMessageWithMetadata extends ChatMessage {
  tools_used?: ToolUsage[];
  citation_dataframe?: any;
  pandas_code?: string;
  cost_info?: any;
}

export default function FinancialReportPage() {
  const [activeTab, setActiveTab] = useState<'chat' | 'report' | 'dashboards'>('chat');
  const [stats, setStats] = useState<FinancialStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [dataLoaded, setDataLoaded] = useState(false);

  // Chat state
  const [chatMessages, setChatMessages] = useState<ChatMessageWithMetadata[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [totalCost, setTotalCost] = useState(0);
  const [queryCount, setQueryCount] = useState(0);

  // Report state
  const [report, setReport] = useState<FinancialReportResponse | null>(null);
  const [isReportGenerating, setIsReportGenerating] = useState(false);
  const [reportProgress, setReportProgress] = useState(0);

  // Visualizations state
  const [figures, setFigures] = useState<any[]>([]);
  const [isVisualizationsLoading, setIsVisualizationsLoading] = useState(false);

  // Load data on mount
  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setIsLoading(true);
      await api.loadFinancialData();
      const statsData = await api.getFinancialStats();
      setStats(statsData);
      setDataLoaded(statsData.loaded);
    } catch (error) {
      console.error('Error loading financial data:', error);
      setDataLoaded(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleChatSubmit = async (message: string) => {
    if (!message.trim() || isChatLoading) return;

    const userMessage: ChatMessageWithMetadata = {
      role: 'user',
      content: message.trim(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsChatLoading(true);

    try {
      const response = await api.financialChat({
        message: message.trim(),
        chat_history: chatMessages.map(msg => ({ role: msg.role, content: msg.content })),
      });

      const assistantMessage: ChatMessageWithMetadata = {
        role: 'assistant',
        content: response.response,
        tools_used: response.tools_used,
        citation_dataframe: response.citation_dataframe,
        pandas_code: response.pandas_code,
        cost_info: response.cost_info,
      };

      setChatMessages(prev => [...prev, assistantMessage]);
      setTotalCost(prev => prev + response.cost_info.total_cost);
      setQueryCount(prev => prev + 1);
    } catch (error) {
      console.error('Error in chat:', error);
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
      }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const generateReport = async () => {
    setIsReportGenerating(true);
    setReportProgress(0);

    try {
      // Simulate progress
      setReportProgress(20);
      await new Promise(resolve => setTimeout(resolve, 300));

      setReportProgress(40);
      const reportData = await api.generateFinancialReport();

      setReportProgress(70);
      await new Promise(resolve => setTimeout(resolve, 200));

      setReport(reportData);
      setTotalCost(prev => prev + reportData.cost_info.total_cost);
      setQueryCount(prev => prev + 1);

      setReportProgress(100);
      await new Promise(resolve => setTimeout(resolve, 300));
    } catch (error) {
      console.error('Error generating report:', error);
    } finally {
      setIsReportGenerating(false);
      setReportProgress(0);
    }
  };

  const loadVisualizations = async () => {
    setIsVisualizationsLoading(true);
    try {
      const response = await api.getFinancialVisualizations();
      const parsedFigures = response.figures.map(figJson => JSON.parse(figJson));
      setFigures(parsedFigures);
    } catch (error) {
      console.error('Error loading visualizations:', error);
    } finally {
      setIsVisualizationsLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab === 'dashboards' && figures.length === 0 && dataLoaded) {
      loadVisualizations();
    }
  }, [activeTab, dataLoaded]);

  const suggestedPrompts = [
    "What were the top 5 products by revenue?",
    "Show me the monthly revenue trend",
    "Which customer segment was most profitable?",
    "Forecast Q4 revenue",
    "What products are declining in revenue?",
    "Show revenue by product category",
    "Which products have the highest profit margin?",
    "What is the average revenue per transaction?",
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-600 to-green-700 rounded-xl p-8 text-white shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-4 mb-4">
              <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
                <DollarSign className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">AI Financial Analyst</h1>
                <p className="text-green-100 mt-1">
                  Powered by LangChain & Gemini 2.5 Flash
                </p>
              </div>
            </div>
          </div>

          {dataLoaded && stats && (
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-sm text-green-100">Transactions</div>
                <div className="text-xl font-bold">{stats.row_count?.toLocaleString()}</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-sm text-green-100">Total Revenue</div>
                <div className="text-xl font-bold">{formatCurrency(stats.total_revenue || 0)}</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-sm text-green-100">Products</div>
                <div className="text-xl font-bold">{stats.unique_products}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Cost Tracking */}
      {dataLoaded && queryCount > 0 && (
        <Card className="bg-gradient-to-r from-blue-50 to-blue-100 border-blue-200">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6">
                <div>
                  <div className="text-sm text-gray-600">Session Queries</div>
                  <div className="text-2xl font-bold text-gray-900">{queryCount}</div>
                </div>
                <div>
                  <div className="text-sm text-gray-600">Total Cost</div>
                  <div className="text-2xl font-bold text-green-600">${totalCost.toFixed(6)}</div>
                </div>
                {queryCount > 0 && (
                  <div>
                    <div className="text-sm text-gray-600">Avg per Query</div>
                    <div className="text-2xl font-bold text-gray-900">${(totalCost / queryCount).toFixed(6)}</div>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {!dataLoaded ? (
        <Card>
          <CardContent className="p-12">
            <div className="text-center">
              <Database className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Data Not Loaded</h3>
              <p className="text-gray-600 mb-4">Click the button below to load sales data</p>
              <Button onClick={loadData} className="flex items-center space-x-2">
                <Database className="w-4 h-4" />
                <span>Load Sales Data</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          {/* Tabs */}
          <div className="border-b border-gray-200">
            <div className="flex space-x-1">
              {[
                { id: 'chat', label: 'Chat with AI', icon: Bot },
                { id: 'report', label: 'AI-Generated Report', icon: BarChart3 },
                { id: 'dashboards', label: 'Data Dashboards', icon: TrendingUp },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={cn(
                      'flex items-center space-x-2 px-6 py-3 font-medium text-sm border-b-2 transition-colors',
                      activeTab === tab.id
                        ? 'border-green-600 text-green-600'
                        : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
                    )}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Tab Content */}
          {activeTab === 'chat' && (
            <ChatTab
              messages={chatMessages}
              input={chatInput}
              setInput={setChatInput}
              onSubmit={handleChatSubmit}
              isLoading={isChatLoading}
              suggestedPrompts={suggestedPrompts}
            />
          )}

          {activeTab === 'report' && (
            <ReportTab
              report={report}
              isGenerating={isReportGenerating}
              progress={reportProgress}
              onGenerate={generateReport}
            />
          )}

          {activeTab === 'dashboards' && (
            <DashboardsTab
              figures={figures}
              isLoading={isVisualizationsLoading}
              onLoad={loadVisualizations}
            />
          )}
        </>
      )}
    </div>
  );
}

function ChatTab({ messages, input, setInput, onSubmit, isLoading, suggestedPrompts }: any) {
  const [expandedTools, setExpandedTools] = useState<Set<number>>(new Set());
  const [expandedCitations, setExpandedCitations] = useState<Set<number>>(new Set());

  const toggleTools = (index: number) => {
    setExpandedTools(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  const toggleCitations = (index: number) => {
    setExpandedCitations(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  const latestMessages = messages.slice(-2);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-xl">Ask the AI Analyst Anything</CardTitle>
          <p className="text-sm text-gray-600">The AI will use tools to retrieve data and cite its sources</p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Messages */}
          {latestMessages.length > 0 ? (
            latestMessages.map((msg, idx) => (
              <div key={idx}>
                {msg.role === 'user' ? (
                  <div className="bg-blue-50 border-l-4 border-blue-600 p-4 rounded-r-lg">
                    <div className="flex items-start space-x-3">
                      <User className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <div className="font-semibold text-gray-900">You:</div>
                        <div className="text-gray-700 mt-1">{msg.content}</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Tools Used */}
                    {msg.tools_used && msg.tools_used.length > 0 && (
                      <div>
                        <button
                          onClick={() => toggleTools(idx)}
                          className="flex items-center justify-between w-full px-4 py-2 bg-yellow-50 border border-yellow-200 rounded-lg hover:bg-yellow-100 transition-colors"
                        >
                          <div className="flex items-center space-x-2">
                            <Code className="w-4 h-4 text-yellow-700" />
                            <span className="text-sm font-medium text-yellow-900">
                              Tools Used ({msg.tools_used.length})
                            </span>
                          </div>
                          {expandedTools.has(idx) ? (
                            <ChevronUp className="w-4 h-4 text-yellow-700" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-yellow-700" />
                          )}
                        </button>

                        {expandedTools.has(idx) && (
                          <div className="mt-2 space-y-2">
                            {msg.tools_used.map((tool: ToolUsage, toolIdx: number) => (
                              <div key={toolIdx} className="bg-yellow-50 border border-yellow-200 p-3 rounded-lg">
                                <div className="text-sm font-mono">
                                  <div className="font-semibold text-yellow-900">🛠️ {tool.tool}</div>
                                  <div className="text-xs text-gray-600 mt-1">
                                    <span className="font-semibold">Input:</span> {tool.input}
                                  </div>
                                  <div className="text-xs text-gray-600 mt-1">
                                    <span className="font-semibold">Output:</span>
                                    <pre className="mt-1 whitespace-pre-wrap">{tool.output.substring(0, 200)}...</pre>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* AI Response */}
                    <div className="bg-gray-50 border-l-4 border-green-600 p-4 rounded-r-lg">
                      <div className="flex items-start space-x-3">
                        <Bot className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <div className="font-semibold text-gray-900">AI Analyst:</div>
                          <div className="prose prose-sm max-w-none mt-1 prose-headings:font-bold prose-headings:text-gray-900 prose-p:text-gray-900 prose-a:text-green-600 prose-code:text-green-600 prose-code:bg-green-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-strong:text-gray-900 prose-li:text-gray-900 [&>*]:text-gray-900">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              rehypePlugins={[rehypeHighlight, rehypeRaw]}
                            >
                              {msg.content}
                            </ReactMarkdown>
                          </div>

                          {msg.cost_info && (
                            <div className="mt-3 text-xs text-gray-500">
                              💰 Cost: ${msg.cost_info.total_cost.toFixed(6)}
                              (Input: {msg.cost_info.input_tokens.toLocaleString()} tokens,
                              Output: {msg.cost_info.output_tokens.toLocaleString()} tokens)
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Citations */}
                    {msg.citation_dataframe && (
                      <div>
                        <button
                          onClick={() => toggleCitations(idx)}
                          className="flex items-center justify-between w-full px-4 py-2 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors"
                        >
                          <div className="flex items-center space-x-2">
                            <Database className="w-4 h-4 text-green-700" />
                            <span className="text-sm font-medium text-green-900">
                              View Source Data (Aggregated)
                            </span>
                          </div>
                          {expandedCitations.has(idx) ? (
                            <ChevronUp className="w-4 h-4 text-green-700" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-green-700" />
                          )}
                        </button>

                        {expandedCitations.has(idx) && (
                          <div className="mt-2 bg-white border border-green-200 rounded-lg p-4 overflow-x-auto">
                            <p className="text-xs text-gray-600 mb-2">
                              📚 This table shows the aggregated data that supports the AI's answer
                            </p>
                            <table className="min-w-full text-sm">
                              <thead>
                                <tr className="border-b">
                                  {msg.citation_dataframe.columns.map((col: string, i: number) => (
                                    <th key={i} className="px-3 py-2 text-left font-medium text-gray-900">{col}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {msg.citation_dataframe.data.map((row: any[], rowIdx: number) => (
                                  <tr key={rowIdx} className="border-b hover:bg-gray-50">
                                    {row.map((cell, cellIdx) => (
                                      <td key={cellIdx} className="px-3 py-2 text-gray-700">
                                        {typeof cell === 'number' ? cell.toLocaleString() : cell}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          ) : (
            <div className="text-center py-12">
              <Bot className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <p className="text-gray-600">Start a conversation by asking a question below</p>
            </div>
          )}

          {isLoading && (
            <div className="bg-gray-50 border-l-4 border-green-600 p-4 rounded-r-lg">
              <div className="flex items-start space-x-3">
                <Bot className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5 animate-pulse" />
                <div>
                  <div className="font-semibold text-gray-900">AI Analyst:</div>
                  <LoadingDots className="mt-2" />
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Input */}
      <Card>
        <CardContent className="p-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              onSubmit(input);
            }}
            className="flex space-x-2"
          >
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="e.g., What were our top products in September?"
              disabled={isLoading}
              className="flex-1 text-gray-900"
            />
            <Button type="submit" disabled={isLoading || !input.trim()}>
              <Send className="w-4 h-4 mr-2" />
              Send
            </Button>
          </form>

          {/* Suggested Prompts */}
          <div className="mt-4">
            <h5 className="text-sm font-medium text-gray-700 mb-2">Or try one of these:</h5>
            <div className="grid grid-cols-2 gap-2">
              {suggestedPrompts.map((prompt: string, idx: number) => (
                <button
                  key={idx}
                  onClick={() => setInput(prompt)}
                  className="px-3 py-2 text-xs text-left bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors text-gray-900"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function ReportTab({ report, isGenerating, progress, onGenerate }: any) {
  const [expandedSteps, setExpandedSteps] = useState(false);

  return (
    <div className="space-y-6">
      {!report && !isGenerating ? (
        <Card>
          <CardContent className="p-12 text-center">
            <BarChart3 className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Generate Comprehensive Report</h3>
            <p className="text-gray-600 mb-6">Let the AI analyze all your data and generate insights</p>
            <Button onClick={onGenerate} size="lg">
              <BarChart3 className="w-5 h-5 mr-2" />
              Generate Report
            </Button>
          </CardContent>
        </Card>
      ) : isGenerating ? (
        <Card>
          <CardContent className="p-12 text-center">
            <Loader2 className="w-16 h-16 mx-auto mb-4 text-green-600 animate-spin" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Generating Report...</h3>
            <div className="w-full max-w-md mx-auto bg-gray-200 rounded-full h-2 mb-4">
              <div
                className="bg-green-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-gray-600">
              {progress < 40 ? 'Analyzing sales data...' :
               progress < 70 ? 'Gathering insights...' :
               'Finalizing report...'}
            </p>
          </CardContent>
        </Card>
      ) : report && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>📄 AI-Generated Financial Report</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="prose prose-lg max-w-none prose-headings:font-bold prose-headings:text-gray-900 prose-p:text-gray-900 prose-a:text-green-600 prose-strong:text-gray-900 prose-li:text-gray-900 prose-code:text-green-600 prose-code:bg-green-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-900 prose-pre:text-gray-100 [&>*]:text-gray-900">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight, rehypeRaw]}
                >
                  {report.response}
                </ReactMarkdown>
              </div>

              {report.cost_info && (
                <div className="mt-6 pt-6 border-t text-sm text-gray-600">
                  💰 Report Generation Cost: ${report.cost_info.total_cost.toFixed(6)} |
                  Input: {report.cost_info.input_tokens.toLocaleString()} tokens |
                  Output: {report.cost_info.output_tokens.toLocaleString()} tokens
                </div>
              )}
            </CardContent>
          </Card>

          {/* Tool Usage */}
          {report.intermediate_steps && report.intermediate_steps.length > 0 && (
            <Card>
              <CardHeader>
                <button
                  onClick={() => setExpandedSteps(!expandedSteps)}
                  className="flex items-center justify-between w-full"
                >
                  <CardTitle className="text-lg">🔧 View AI's Work</CardTitle>
                  {expandedSteps ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                </button>
              </CardHeader>
              {expandedSteps && (
                <CardContent className="space-y-4">
                  {report.intermediate_steps.map((step: ToolUsage, idx: number) => (
                    <div key={idx} className="border-l-4 border-green-500 bg-gray-50 p-4 rounded-r-lg">
                      <div className="font-semibold text-gray-900">Step {idx + 1}: {step.tool}</div>
                      <div className="text-sm text-gray-600 mt-2">
                        <div className="font-medium">Input:</div>
                        <code className="block mt-1 text-xs bg-white p-2 rounded">{step.input}</code>
                      </div>
                      <div className="text-sm text-gray-600 mt-2">
                        <div className="font-medium">Output:</div>
                        <pre className="block mt-1 text-xs bg-white p-2 rounded overflow-x-auto">
                          {step.output.substring(0, 300)}...
                        </pre>
                      </div>
                    </div>
                  ))}
                </CardContent>
              )}
            </Card>
          )}

          <div className="flex justify-center">
            <Button onClick={onGenerate} variant="outline">
              🔄 Generate New Report
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

function DashboardsTab({ figures, isLoading, onLoad }: any) {
  if (figures.length === 0 && !isLoading) {
    return (
      <Card>
        <CardContent className="p-12 text-center">
          <TrendingUp className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Generate Visualizations</h3>
          <p className="text-gray-600 mb-6">Create interactive charts and dashboards</p>
          <Button onClick={onLoad} size="lg">
            <BarChart3 className="w-5 h-5 mr-2" />
            Generate Dashboards
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-12 text-center">
          <Loader2 className="w-16 h-16 mx-auto mb-4 text-green-600 animate-spin" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Creating Visualizations...</h3>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {figures.map((fig: any, idx: number) => (
        <Card key={idx}>
          <CardContent className="p-6">
            <Plot
              data={fig.data}
              layout={{ ...fig.layout, autosize: true }}
              config={{ responsive: true }}
              className="w-full"
              style={{ width: '100%', height: '500px' }}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
