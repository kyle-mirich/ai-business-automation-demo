'use client';

import { useState, useEffect } from 'react';
import { Upload, Package, AlertTriangle, TrendingUp, TrendingDown, DollarSign, BarChart3, Sparkles } from 'lucide-react';
import { api } from '@/lib/api';
import type { InventoryAnalysisResponse, InventoryItem } from '@/lib/types';
import { Button } from '@/components/ui/Button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { getStatusColor, formatCurrency, formatNumber } from '@/lib/utils';

export default function InventoryOptimizerPage() {
  const [result, setResult] = useState<InventoryAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<keyof InventoryItem>('current_stock');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Auto-run analysis on component mount for demo purposes
  useEffect(() => {
    analyzeDefault();
  }, []);

  const analyzeDefault = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.analyzeInventory();
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.uploadInventory(file);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const sortedItems = result?.items.slice().sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];

    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    }

    return sortDirection === 'asc'
      ? String(aVal).localeCompare(String(bVal))
      : String(bVal).localeCompare(String(aVal));
  });

  const toggleSort = (field: keyof InventoryItem) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-600 to-orange-700 rounded-xl p-8 text-white shadow-lg">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-4 mb-4">
              <div className="bg-white/20 p-3 rounded-lg backdrop-blur-sm">
                <Package className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold">Inventory Optimizer</h1>
                <p className="text-orange-100 mt-1">
                  Powered by Prophet ML & LangChain Agents
                </p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-xl font-bold">Prophet ML</div>
                <div className="text-sm text-orange-100">Demand Forecasting</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-xl font-bold">Smart Alerts</div>
                <div className="text-sm text-orange-100">Stock Optimization</div>
              </div>
              <div className="bg-white/10 rounded-lg p-3 backdrop-blur-sm">
                <div className="text-xl font-bold">AI Insights</div>
                <div className="text-sm text-orange-100">Auto Recommendations</div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-3">
            <Button
              onClick={analyzeDefault}
              disabled={isLoading}
              className="bg-white text-orange-600 hover:bg-orange-50"
            >
              {isLoading ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Analyzing...
                </>
              ) : (
                <>
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Re-Analyze
                </>
              )}
            </Button>
            <label className="cursor-pointer">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                disabled={isLoading}
              />
              <div className="btn-secondary bg-white/20 hover:bg-white/30 text-white border-white/30 flex items-center justify-center space-x-2 px-4 py-2 rounded-lg">
                <Upload className="w-4 h-4" />
                <span>Upload CSV</span>
              </div>
            </label>
          </div>
        </div>
      </div>

      {error && (
        <Card className="bg-red-50 border-red-200">
          <CardContent className="p-4">
            <p className="text-sm text-red-700">{error}</p>
          </CardContent>
        </Card>
      )}

      {result && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card className="p-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                  <Package className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total SKUs</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {result.summary.total_skus || 0}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-red-100 flex items-center justify-center">
                  <AlertTriangle className="w-5 h-5 text-red-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Low Stock</p>
                  <p className="text-2xl font-bold text-red-600">
                    {result.summary.low_stock_count || 0}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-orange-100 flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-orange-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Overstock</p>
                  <p className="text-2xl font-bold text-orange-600">
                    {result.summary.overstock_count || 0}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded-lg bg-green-100 flex items-center justify-center">
                  <DollarSign className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Value</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatCurrency(result.summary.total_inventory_value || 0)}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          {/* AI Insights */}
          {result.ai_insights && (
            <Card>
              <CardHeader>
                <CardTitle>AI Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-gray-700 whitespace-pre-wrap">{result.ai_insights}</p>
              </CardContent>
            </Card>
          )}

          {/* Recommendations */}
          {result.recommendations && result.recommendations.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Top Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {result.recommendations.slice(0, 10).map((rec, index) => (
                    <div
                      key={index}
                      className="border-l-4 border-primary-500 bg-gray-50 p-4 rounded-r-lg"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <Badge
                            variant={
                              rec.urgency.toLowerCase() === 'high' ? 'error' :
                              rec.urgency.toLowerCase() === 'medium' ? 'warning' : 'default'
                            }
                          >
                            {rec.urgency}
                          </Badge>
                          <span className="font-medium text-gray-900">
                            {rec.product_name} ({rec.sku})
                          </span>
                        </div>
                        {rec.estimated_cost_impact && (
                          <span className="text-sm font-medium text-gray-600">
                            {formatCurrency(rec.estimated_cost_impact)}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-700 mb-1">{rec.action}</p>
                      <p className="text-xs text-gray-500">{rec.reasoning}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Inventory Table */}
          <Card>
            <CardHeader>
              <CardTitle>Inventory Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200">
                      <TableHeader field="sku" label="SKU" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="product_name" label="Product" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="status" label="Status" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="current_stock" label="Stock" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="reorder_point" label="Reorder Point" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="last_30_days_sales" label="30d Sales" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="forecast_30d" label="Forecast" onSort={toggleSort} current={sortField} direction={sortDirection} />
                      <TableHeader field="cost_per_unit" label="Cost" onSort={toggleSort} current={sortField} direction={sortDirection} />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {sortedItems?.map((item, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="py-3 px-2 font-mono text-xs">{item.sku}</td>
                        <td className="py-3 px-2 font-medium">{item.product_name}</td>
                        <td className="py-3 px-2">
                          <Badge className={getStatusColor(item.status)}>
                            {item.status.replace(/_/g, ' ')}
                          </Badge>
                        </td>
                        <td className="py-3 px-2 text-right">{formatNumber(item.current_stock)}</td>
                        <td className="py-3 px-2 text-right">{formatNumber(item.reorder_point)}</td>
                        <td className="py-3 px-2 text-right">{formatNumber(item.last_30_days_sales)}</td>
                        <td className="py-3 px-2 text-right">
                          {item.forecast_30d ? formatNumber(Math.round(item.forecast_30d)) : '-'}
                        </td>
                        <td className="py-3 px-2 text-right">{formatCurrency(item.cost_per_unit)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 text-sm text-gray-500 text-center">
                Showing {sortedItems?.length || 0} items
              </div>
            </CardContent>
          </Card>

          {/* Metrics */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card className="p-4">
              <p className="text-sm text-gray-500">Stockout Risk</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                {result.summary.stockout_risk_pct?.toFixed(1) || 0}%
              </p>
            </Card>

            <Card className="p-4">
              <p className="text-sm text-gray-500">Avg Lead Time</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                {result.summary.average_lead_time?.toFixed(0) || 0} days
              </p>
            </Card>

            <Card className="p-4">
              <p className="text-sm text-gray-500">API Cost</p>
              <p className="text-2xl font-bold text-gray-900 mt-1">
                {formatCurrency(result.cost_usd)}
              </p>
            </Card>
          </div>
        </>
      )}

      {!result && !isLoading && (
        <Card className="p-12">
          <div className="text-center text-gray-500">
            <Package className="w-16 h-16 mx-auto mb-4 text-gray-300" />
            <p className="text-lg font-medium text-gray-700 mb-2">No Analysis Yet</p>
            <p>Click "Analyze Default Data" or upload a CSV file to get started</p>
          </div>
        </Card>
      )}
    </div>
  );
}

function TableHeader({
  field,
  label,
  onSort,
  current,
  direction,
}: {
  field: keyof InventoryItem;
  label: string;
  onSort: (field: keyof InventoryItem) => void;
  current: keyof InventoryItem;
  direction: 'asc' | 'desc';
}) {
  const isActive = current === field;

  return (
    <th
      onClick={() => onSort(field)}
      className="py-3 px-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-50 select-none"
    >
      <div className="flex items-center space-x-1">
        <span>{label}</span>
        {isActive && (
          <span className="text-primary-600">
            {direction === 'asc' ? '↑' : '↓'}
          </span>
        )}
      </div>
    </th>
  );
}
