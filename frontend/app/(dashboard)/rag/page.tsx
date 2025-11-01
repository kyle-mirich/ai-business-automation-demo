'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Copy, Check, FileText, ExternalLink, ChevronDown, ChevronUp, Eye } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github-dark.css';
import { api } from '@/lib/api';
import type { RAGQueryResponse, SourceDocument } from '@/lib/types';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { LoadingDots } from '@/components/ui/LoadingSpinner';
import { cn } from '@/lib/utils';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDocument[];
  timestamp: Date;
}

export default function RAGChatbotPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingContent('');

    try {
      // Use streaming API
      const stream = api.queryRAGStream({
        query: userMessage.content,
        top_k: 10,
      });

      let fullResponse = '';
      let sources: SourceDocument[] = [];

      for await (const chunk of stream) {
        try {
          const data = JSON.parse(chunk);

          if (data.type === 'token') {
            fullResponse += data.content;
            setStreamingContent(fullResponse);
          } else if (data.type === 'sources') {
            sources = data.content;
          } else if (data.type === 'complete') {
            fullResponse = data.content;
            sources = data.sources || sources;
          } else if (data.type === 'error') {
            throw new Error(data.content);
          }
        } catch (parseError) {
          console.error('Parse error:', parseError);
        }
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: fullResponse || streamingContent,
        sources,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setStreamingContent('');
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-xl p-8 text-white shadow-lg">
        <div className="flex items-center space-x-4 mb-4">
          <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
            <Bot className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">RAG Chatbot</h1>
            <p className="text-primary-100 mt-1">
              Powered by LangChain, ChromaDB & Gemini 2.5 Flash
            </p>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-4 mt-6">
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="text-2xl font-bold">12</div>
            <div className="text-sm text-primary-100">Research Papers</div>
          </div>
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="text-2xl font-bold">Vector Search</div>
            <div className="text-sm text-primary-100">RAG Architecture</div>
          </div>
          <div className="bg-white/10 rounded-lg p-4 backdrop-blur-sm">
            <div className="text-2xl font-bold">Real-time</div>
            <div className="text-sm text-primary-100">Streaming Responses</div>
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <Card className="flex flex-col h-[700px] shadow-xl">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-gradient-to-b from-gray-50 to-white">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="relative mb-6">
                <div className="absolute inset-0 bg-primary-600 opacity-20 blur-3xl rounded-full"></div>
                <Bot className="relative w-20 h-20 text-primary-600" />
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">
                Ask Me Anything About AI Research
              </h3>
              <p className="text-gray-600 max-w-2xl mb-8">
                I have access to 12 cutting-edge research papers on transformers, BERT, GPT, attention mechanisms, and more.
                Ask technical questions and get answers with precise citations.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-3xl w-full">
                {[
                  'What is the attention mechanism in transformers?',
                  'Explain the BERT pre-training process',
                  'How does GPT differ from BERT?',
                  'What are the advantages of transformer architecture?',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="px-6 py-4 text-sm text-left bg-white border-2 border-gray-200 rounded-xl hover:border-primary-400 hover:shadow-md transition-all duration-200 group"
                  >
                    <div className="flex items-start space-x-3">
                      <Send className="w-4 h-4 text-gray-400 group-hover:text-primary-600 transition-colors mt-0.5" />
                      <span className="text-gray-700 group-hover:text-gray-900">{suggestion}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <MessageBubble key={index} message={message} />
          ))}

          {/* Streaming Message */}
          {isLoading && streamingContent && (
            <div className="flex items-start space-x-4 animate-fadeIn">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center flex-shrink-0 shadow-lg">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1 bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight, rehypeRaw]}
                  >
                    {streamingContent}
                  </ReactMarkdown>
                </div>
                <LoadingDots className="mt-3" />
              </div>
            </div>
          )}

          {isLoading && !streamingContent && (
            <div className="flex items-start space-x-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center flex-shrink-0 shadow-lg">
                <Bot className="w-6 h-6 text-white animate-pulse" />
              </div>
              <div className="bg-white rounded-2xl p-6 shadow-lg border border-gray-100">
                <LoadingDots />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        <div className="border-t border-gray-200 p-6 bg-white">
          <form onSubmit={handleSubmit} className="flex space-x-3">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question about AI research papers..."
              disabled={isLoading}
              className="flex-1 h-12 text-base"
            />
            <Button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="h-12 px-6"
            >
              <Send className="w-5 h-5 mr-2" />
              Send
            </Button>
          </form>
          <p className="text-xs text-gray-500 mt-3 text-center">
            Powered by LangChain RAG pipeline with ChromaDB vector search
          </p>
        </div>
      </Card>
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const [copied, setCopied] = useState(false);
  const [sourcesExpanded, setSourcesExpanded] = useState(true);
  const [expandedSourceIndex, setExpandedSourceIndex] = useState<number | null>(null);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isUser = message.role === 'user';

  return (
    <div className={cn('flex items-start space-x-4', isUser && 'flex-row-reverse space-x-reverse')}>
      <div className={cn(
        'w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg',
        isUser
          ? 'bg-gradient-to-br from-gray-700 to-gray-800'
          : 'bg-gradient-to-br from-primary-500 to-primary-600'
      )}>
        {isUser ? (
          <User className="w-6 h-6 text-white" />
        ) : (
          <Bot className="w-6 h-6 text-white" />
        )}
      </div>

      <div className={cn('flex-1 max-w-4xl', isUser && 'flex justify-end')}>
        <div className={cn(
          'rounded-2xl p-6 shadow-lg border',
          isUser
            ? 'bg-gradient-to-br from-gray-700 to-gray-800 text-white border-gray-700'
            : 'bg-white border-gray-100'
        )}>
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              {isUser ? (
                <p className="text-white whitespace-pre-wrap text-base leading-relaxed">
                  {message.content}
                </p>
              ) : (
                <div className="prose prose-sm max-w-none prose-headings:font-bold prose-p:text-gray-700 prose-a:text-primary-600 prose-code:text-primary-600 prose-code:bg-primary-50 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-900 prose-pre:text-gray-100">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeHighlight, rehypeRaw]}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              )}
            </div>
            {!isUser && (
              <button
                onClick={copyToClipboard}
                className="text-gray-400 hover:text-gray-600 transition-colors flex-shrink-0 p-2 rounded-lg hover:bg-gray-100"
                title="Copy to clipboard"
              >
                {copied ? (
                  <Check className="w-5 h-5 text-green-600" />
                ) : (
                  <Copy className="w-5 h-5" />
                )}
              </button>
            )}
          </div>

          {/* Sources Section */}
          {!isUser && message.sources && message.sources.length > 0 && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <button
                onClick={() => setSourcesExpanded(!sourcesExpanded)}
                className="flex items-center justify-between w-full text-left mb-4 group"
              >
                <div className="flex items-center space-x-3">
                  <div className="bg-primary-100 p-2 rounded-lg group-hover:bg-primary-200 transition-colors">
                    <FileText className="w-5 h-5 text-primary-600" />
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900 text-base">
                      Sources & Citations
                    </div>
                    <div className="text-sm text-gray-500">
                      {message.sources.length} research paper excerpts
                    </div>
                  </div>
                </div>
                {sourcesExpanded ? (
                  <ChevronUp className="w-5 h-5 text-gray-400" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-gray-400" />
                )}
              </button>

              {sourcesExpanded && (
                <div className="grid gap-3">
                  {message.sources.map((source, idx) => (
                    <SourceCard
                      key={idx}
                      source={source}
                      index={idx}
                      isExpanded={expandedSourceIndex === idx}
                      onToggle={() => setExpandedSourceIndex(expandedSourceIndex === idx ? null : idx)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          <div className={cn(
            'mt-4 text-xs flex items-center justify-between',
            isUser ? 'text-gray-300' : 'text-gray-500'
          )}>
            <span>{message.timestamp.toLocaleTimeString()}</span>
            {!isUser && message.sources && message.sources.length > 0 && (
              <span className="flex items-center space-x-1">
                <FileText className="w-3 h-3" />
                <span>{message.sources.length} sources</span>
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function SourceCard({
  source,
  index,
  isExpanded,
  onToggle
}: {
  source: SourceDocument;
  index: number;
  isExpanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="bg-gradient-to-br from-gray-50 to-gray-100 border-2 border-gray-200 rounded-xl overflow-hidden hover:border-primary-300 transition-all duration-200">
      <button
        onClick={onToggle}
        className="w-full p-4 text-left flex items-center justify-between hover:bg-white/50 transition-colors"
      >
        <div className="flex items-center space-x-3 flex-1 min-w-0">
          <Badge variant="info" className="flex-shrink-0 text-sm font-bold px-3 py-1">
            {index + 1}
          </Badge>
          <div className="flex-1 min-w-0">
            <div className="font-semibold text-gray-900 truncate text-sm">
              {source.file}
            </div>
            <div className="flex items-center space-x-2 mt-1">
              <Badge variant="default" className="text-xs">
                Page {source.page}
              </Badge>
              {source.score && (
                <span className="text-xs text-gray-600">
                  Relevance: {(source.score * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2 flex-shrink-0 ml-4">
          {source.source_url && (
            <a
              href={source.source_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="p-2 text-primary-600 hover:bg-primary-100 rounded-lg transition-colors"
              title="View PDF"
            >
              <Eye className="w-4 h-4" />
            </a>
          )}
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-gray-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-gray-400" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 space-y-3 bg-white/50">
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">
              Excerpt with Highlights
            </div>
            <div
              className="text-sm text-gray-800 leading-relaxed prose prose-sm max-w-none"
              dangerouslySetInnerHTML={{ __html: source.highlighted_excerpt || source.chunk_excerpt }}
            />
          </div>

          {source.chunk_full && source.chunk_full !== source.chunk_excerpt && (
            <details className="bg-white border border-gray-200 rounded-lg">
              <summary className="px-4 py-3 cursor-pointer text-xs font-semibold text-gray-600 uppercase tracking-wide hover:bg-gray-50 transition-colors">
                View Full Chunk
              </summary>
              <div className="px-4 pb-4 pt-2">
                <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                  {source.chunk_full}
                </p>
              </div>
            </details>
          )}

          {source.metadata && Object.keys(source.metadata).length > 0 && (
            <details className="bg-white border border-gray-200 rounded-lg">
              <summary className="px-4 py-3 cursor-pointer text-xs font-semibold text-gray-600 uppercase tracking-wide hover:bg-gray-50 transition-colors">
                Technical Metadata
              </summary>
              <div className="px-4 pb-4 pt-2">
                <pre className="text-xs text-gray-600 overflow-x-auto">
                  {JSON.stringify(source.metadata, null, 2)}
                </pre>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
