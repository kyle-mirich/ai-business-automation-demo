import Link from 'next/link';
import { Bot, HeadphonesIcon, Package, ArrowRight, Sparkles, DollarSign } from 'lucide-react';

export default function Home() {
  const features = [
    {
      icon: DollarSign,
      title: 'Financial Report',
      description: 'Q3 2025 sales analytics with AI-powered business insights and trend analysis',
      href: '/financial',
      gradient: 'from-green-500 to-emerald-500',
    },
    {
      icon: HeadphonesIcon,
      title: 'Support Triage',
      description: 'Automated ticket classification, prioritization, and AI-generated responses',
      href: '/support',
      gradient: 'from-purple-500 to-pink-500',
    },
    {
      icon: Package,
      title: 'Inventory Optimizer',
      description: 'AI-powered demand forecasting and reorder recommendations with Prophet ML',
      href: '/inventory',
      gradient: 'from-orange-500 to-red-500',
    },
    {
      icon: Bot,
      title: 'RAG Chatbot',
      description: 'Query 12 AI research papers with intelligent retrieval and streaming responses',
      href: '/rag',
      gradient: 'from-blue-500 to-cyan-500',
    },
  ];

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-500/20 to-purple-500/20 blur-3xl" />

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-24">
          <div className="text-center animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-50 border border-primary-200 rounded-full mb-6">
              <Sparkles className="w-4 h-4 text-primary-600" />
              <span className="text-sm font-medium text-primary-900">Powered by Google Gemini 2.5 Flash</span>
            </div>

            <h1 className="text-5xl sm:text-6xl font-bold text-gray-900 mb-6">
              AI Business Automation
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-purple-600">
                Demo Platform
              </span>
            </h1>

            <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
              Enterprise-grade AI solutions for modern businesses. Built with Next.js, FastAPI, and LangChain.
            </p>

            <div className="flex gap-4 justify-center">
              <Link href="/financial" className="btn-primary flex items-center gap-2">
                Get Started <ArrowRight className="w-4 h-4" />
              </Link>
              <a
                href="https://github.com/kyle-mirich/ai-business-automation-demo"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-secondary"
              >
                View on GitHub
              </a>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mt-20">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <Link
                  key={feature.href}
                  href={feature.href}
                  className="group card p-6 hover:scale-105 transition-transform"
                >
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>

                  <h3 className="text-xl font-bold text-gray-900 mb-2 group-hover:text-primary-600 transition-colors">
                    {feature.title}
                  </h3>

                  <p className="text-gray-600 mb-4">
                    {feature.description}
                  </p>

                  <div className="flex items-center text-primary-600 font-medium group-hover:gap-2 transition-all">
                    Try it now <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </Link>
              );
            })}
          </div>

          {/* Tech Stack */}
          <div className="mt-20 text-center">
            <p className="text-sm font-medium text-gray-500 mb-4">BUILT WITH</p>
            <div className="flex flex-wrap justify-center gap-6 text-gray-700">
              <span className="font-medium">Next.js 15</span>
              <span className="text-gray-300">•</span>
              <span className="font-medium">TypeScript</span>
              <span className="text-gray-300">•</span>
              <span className="font-medium">Tailwind CSS</span>
              <span className="text-gray-300">•</span>
              <span className="font-medium">FastAPI</span>
              <span className="text-gray-300">•</span>
              <span className="font-medium">LangChain</span>
              <span className="text-gray-300">•</span>
              <span className="font-medium">ChromaDB</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
