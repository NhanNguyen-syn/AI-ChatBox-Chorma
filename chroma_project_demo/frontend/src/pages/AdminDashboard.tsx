import React, { useState, useEffect } from 'react'
import { Users, MessageSquare, Settings, BarChart3, Palette } from 'lucide-react'
import { api } from '../services/api'
import toast from 'react-hot-toast'
import tinycolor from 'tinycolor2'
import { useBranding } from '../contexts/BrandingContext'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface User {
    id: number
    username: string
    email: string
    full_name: string
    is_admin: boolean
    is_active: boolean
    created_at: string
    chat_count: number
}

interface ChatStats {
    total_sessions: number
    total_messages: number
    total_tokens: number
    avg_response_time: number
    active_users_today: number
    active_users_week: number
}



type TabId = 'overview' | 'users' | 'config' | 'branding'

interface AdminDashboardProps {
    initialTab?: TabId
}

const AdminDashboard: React.FC<AdminDashboardProps> = ({ initialTab = 'overview' }) => {
    const [activeTab, setActiveTab] = useState<TabId>(initialTab)
    const { brandingConfig: contextBrandingConfig, loadBrandingConfig: reloadBranding } = useBranding()
    const [users, setUsers] = useState<User[]>([])
    const [stats, setStats] = useState<ChatStats | null>(null)

    const [loading, setLoading] = useState(true)
    const [activity, setActivity] = useState<any[]>([])
    // const [showCreate, setShowCreate] = useState(false)
    // const [form, setForm] = useState({ question: '', answer: '', category: '' })
    const [config, setConfig] = useState({
        chat_model: 'gpt-4o-mini',
        embed_model: 'text-embedding-3-small',
        similarity_threshold: 0.7,
        max_tokens: 500,
    })

    const loadConfigs = async () => {
        try {
            const res = await api.get('/admin/system-configs')
            const list = res.data || []
            const map: any = {}
            list.forEach((c: any) => { map[c.key] = c.value })
            setConfig((prev) => ({
                ...prev,
                chat_model: map['chat_model'] ?? prev.chat_model,
                embed_model: map['embed_model'] ?? prev.embed_model,
                similarity_threshold: map['similarity_threshold'] ? parseFloat(map['similarity_threshold']) : prev.similarity_threshold,
                max_tokens: map['max_tokens'] ? parseInt(map['max_tokens']) : prev.max_tokens,
            }))
        } catch (e) {
            // ignore if not configured yet
        }
    }

    const saveConfigs = async () => {
        try {
            await Promise.all([
                api.put('/admin/system-configs', { key: 'chat_model', value: String(config.chat_model), description: 'chat model' }),
                api.put('/admin/system-configs', { key: 'embed_model', value: String(config.embed_model), description: 'embed model' }),
                api.put('/admin/system-configs', { key: 'similarity_threshold', value: String(config.similarity_threshold), description: 'similarity threshold' }),
                api.put('/admin/system-configs', { key: 'max_tokens', value: String(config.max_tokens), description: 'max tokens' }),
            ])
            toast.success('Đã lưu cấu hình')
        } catch (e: any) {
            toast.error(e?.response?.data?.detail || 'Lưu cấu hình thất bại')
        }
    }

    const [brandingConfig, setBrandingConfig] = useState({
        brand_name: '',
        primary_color: '#1FAA59',
        brand_logo_url: '',
        brand_logo_height: '32px',
        favicon_url: ''
    });

    useEffect(() => {
        if (contextBrandingConfig) {
            setBrandingConfig(contextBrandingConfig);
        }
    }, [contextBrandingConfig]);

    const saveBrandingConfig = async () => {
        try {
            await api.put('/admin/branding', brandingConfig);
            toast.success('Đã lưu cấu hình thương hiệu');
            reloadBranding();
        } catch (e: any) {
            toast.error(e?.response?.data?.detail || 'Lưu cấu hình thất bại');
        }
    };

    const handleBrandingFileUpload = async (type: 'logo' | 'favicon', file: File) => {
        if (!file) return;
        const endpoint = type === 'logo' ? '/admin/branding/upload-logo' : '/admin/branding/upload-favicon';
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await api.post(endpoint, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            if (type === 'logo') {
                setBrandingConfig(prev => ({ ...prev, brand_logo_url: res.data.url }));
            } else {
                setBrandingConfig(prev => ({ ...prev, favicon_url: res.data.url }));
            }
            toast.success('Tải file lên thành công!');
        } catch (e: any) {
            toast.error(e?.response?.data?.detail || 'Tải file lên thất bại');
        }
    };

    useEffect(() => {
        loadData()
        loadConfigs()
    }, [])

    const loadData = async () => {
        setLoading(true)
        try {
            // Load each endpoint separately to see which one fails
            console.log('Loading users...')
            const usersRes = await api.get('/admin/users')
            console.log('Users loaded:', usersRes.data)
            setUsers(usersRes.data ?? [])

            console.log('Loading stats...')
            const statsRes = await api.get('/admin/stats')
            console.log('Stats loaded:', statsRes.data)
            setStats(statsRes.data ?? null)



            console.log('Loading activity...')
            const activityRes = await api.get('/admin/activity')
            console.log('Activity loaded:', activityRes.data)
            setActivity(activityRes.data ?? [])

            console.log('All data loaded successfully')
        } catch (error: any) {
            console.error('Load admin data error:', error?.response?.data || error?.message)
            console.error('Full error:', error)
            toast.error(`Không thể tải dữ liệu: ${error?.response?.status || error?.message}`)
        } finally {
            setLoading(false)
        }
    }

    const toggleUserStatus = async (userId: number) => {
        try {
            await api.put(`/admin/users/${userId}/toggle`)
            setUsers(users.map(user =>
                user.id === userId ? { ...user, is_active: !user.is_active } : user
            ))
            toast.success('Cập nhật trạng thái user thành công')
        } catch (error) {
            toast.error('Không thể cập nhật trạng thái user')
        }
    }

    const toggleAdminStatus = async (userId: number) => {
        try {
            await api.put(`/admin/users/${userId}/admin`)
            setUsers(users.map(user =>
                user.id === userId ? { ...user, is_admin: !user.is_admin } : user
            ))
            toast.success('Cập nhật quyền admin thành công')
        } catch (error) {
            toast.error('Không thể cập nhật quyền admin')
        }
    }



    const tabs: { id: TabId; name: string; icon: any }[] = [
        { id: 'overview', name: 'Tổng quan', icon: BarChart3 },
        { id: 'users', name: 'Quản lý User', icon: Users },


        { id: 'branding', name: 'Thương hiệu', icon: Palette },
    ]

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
                <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Admin Dashboard</h1>
                <p className="text-gray-600 dark:text-gray-400 mt-2">Quản lý hệ thống Chroma AI Chat</p>
            </div>

            {/* Navigation Tabs */}
            <div className="bg-white rounded-lg shadow dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
                <div className="border-b border-gray-200 dark:border-gray-800">
                    <nav className="flex space-x-8 px-6">
                        {tabs.map((tab) => {
                            const Icon = tab.icon
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${activeTab === tab.id
                                        ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:border-gray-700'
                                        }`}
                                >
                                    <Icon className="h-5 w-5" />
                                    <span>{tab.name}</span>
                                </button>
                            )
                        })}
                    </nav>
                </div>

                <div className="p-6">
                    {/* Overview Tab */}
                    {activeTab === 'overview' && stats && (
                        <div className="space-y-6">
                            {/* Stats Cards */}
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                                <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <MessageSquare className="h-8 w-8 text-blue-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Tổng tin nhắn</p>
                                            <p className="text-2xl font-semibold text-blue-900 dark:text-blue-200">{stats.total_messages}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-green-50 dark:bg-green-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <Users className="h-8 w-8 text-green-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-green-600 dark:text-green-400">User hoạt động (tuần)</p>
                                            <p className="text-2xl font-semibold text-green-900 dark:text-green-200">{stats.active_users_week}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-yellow-50 dark:bg-yellow-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <BarChart3 className="h-8 w-8 text-yellow-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-yellow-600 dark:text-yellow-400">Tổng tokens</p>
                                            <p className="text-2xl font-semibold text-yellow-900 dark:text-yellow-200">{stats.total_tokens.toLocaleString()}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-purple-50 dark:bg-purple-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <Settings className="h-8 w-8 text-purple-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-purple-600 dark:text-purple-400">Thời gian phản hồi TB</p>
                                            <p className="text-2xl font-semibold text-purple-900 dark:text-purple-200">{stats.avg_response_time.toFixed(2)}s</p>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Chart */}
                            <div className="bg-white rounded-lg p-6 border dark:bg-[#0f0f0f] dark:border-gray-800">
                                <h3 className="text-lg font-semibold mb-4">Hoạt động chat (7 ngày qua)</h3>
                                <div className="h-64">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={activity.map((x) => ({ day: new Date(x.date).toLocaleDateString('vi-VN', { weekday: 'short' }), messages: x.count }))}>
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="day" />
                                            <YAxis allowDecimals={false} domain={[0, 100]} ticks={[25, 50, 75, 100]} />
                                            <Tooltip contentStyle={{ backgroundColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#0f0f0f' : '#ffffff', borderColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#1f2937' : '#e5e7eb', color: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#e5e7eb' : '#111827' }} />
                                            <Line type="monotone" dataKey="messages" stroke="#3b82f6" strokeWidth={2} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Users Tab */}
                    {activeTab === 'users' && (
                        <div className="space-y-4">
                            <div className="flex justify-between items-center">
                                <h3 className="text-lg font-semibold">Danh sách User</h3>
                                <span className="text-sm text-gray-500 dark:text-gray-400">Tổng: {users.length} users</span>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
                                    <thead className="bg-gray-50 dark:bg-[#0f0f0f]">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">
                                                User
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">
                                                Email
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">
                                                Chat count
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">
                                                Trạng thái
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">
                                                Quyền
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">
                                                Hành động
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody className="bg-white divide-y divide-gray-200 dark:bg-[#0b0b0b] dark:divide-gray-800">
                                        {users.map((user) => (
                                            <tr key={user.id}>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <div>
                                                        <div className="text-sm font-medium text-gray-900 dark:text-gray-100">{user.full_name}</div>
                                                        <div className="text-sm text-gray-500 dark:text-gray-400">@{user.username}</div>
                                                    </div>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                                                    {user.email}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
                                                    {user.chat_count}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${user.is_active
                                                        ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300'
                                                        : 'bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300'
                                                        }`}>
                                                        {user.is_active ? 'Hoạt động' : 'Bị khóa'}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${user.is_admin
                                                        ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300'
                                                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                                                        }`}>
                                                        {user.is_admin ? 'Admin' : 'User'}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                                                    <button
                                                        onClick={() => toggleUserStatus(user.id)}
                                                        className="text-blue-600 hover:text-blue-900"
                                                    >
                                                        {user.is_active ? 'Khóa' : 'Mở khóa'}
                                                    </button>
                                                    <button
                                                        onClick={() => toggleAdminStatus(user.id)}
                                                        className="text-purple-600 hover:text-purple-900"
                                                    >
                                                        {user.is_admin ? 'Bỏ admin' : 'Thêm admin'}
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    )}

                    {/* FAQs Tab (đã ẩn) */}

                    {/* Config Tab */}
                    {activeTab === 'config' && (
                        <div className="space-y-6">
                            <h3 className="text-lg font-semibold">Cấu hình AI</h3>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Mô hình AI</h4>
                                    <select className="input-field" value={config.chat_model} onChange={(e) => setConfig((c) => ({ ...c, chat_model: e.target.value }))}>
                                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                                        <option value="gpt-4">GPT-4</option>
                                        <option value="gpt-4-turbo">GPT-4 Turbo</option>
                                        <option value="gpt-4o-mini">gpt-4o-mini</option>
                                    </select>
                                </div>

                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Embedding Model</h4>
                                    <select className="input-field" value={config.embed_model} onChange={(e) => setConfig((c) => ({ ...c, embed_model: e.target.value }))}>
                                        <option value="text-embedding-3-small">text-embedding-3-small</option>
                                        <option value="text-embedding-3-large">text-embedding-3-large</option>
                                    </select>
                                </div>

                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Ngưỡng Similarity</h4>
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.1"
                                        value={config.similarity_threshold}
                                        onChange={(e) => setConfig((c) => ({ ...c, similarity_threshold: parseFloat(e.target.value) }))}
                                        className="w-full"
                                    />
                                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                                        <span>0.0</span>
                                        <span>{config.similarity_threshold.toFixed(2)}</span>
                                        <span>1.0</span>
                                    </div>
                                </div>

                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Max Tokens</h4>
                                    <input
                                        type="number"
                                        value={config.max_tokens}
                                        onChange={(e) => setConfig((c) => ({ ...c, max_tokens: parseInt(e.target.value || '0') }))}
                                        className="input-field"
                                        min="100"
                                        max="128000"
                                    />
                                </div>
                            </div>

                            <div className="flex justify-end">
                                <button className="btn-primary" onClick={saveConfigs}>
                                    Lưu cấu hình
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Branding Tab */}
                    {activeTab === 'branding' && (
                        <div className="space-y-6">
                            <h3 className="text-lg font-semibold">Cấu hình Thương hiệu</h3>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Tên thương hiệu</h4>
                                    <input
                                        type="text"
                                        value={brandingConfig.brand_name}
                                        onChange={(e) => setBrandingConfig(c => ({ ...c, brand_name: e.target.value }))}
                                        className="input-field"
                                        placeholder="e.g., Chroma AI Chat"
                                    />
                                </div>

                                <div className="bg-gray-50 rounded-lg p-4">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Màu chủ đề</h4>
                                    <input
                                        type="color"
                                        value={brandingConfig.primary_color}
                                        onChange={(e) => {
                                            const newColor = e.target.value;
                                            setBrandingConfig(c => ({ ...c, primary_color: newColor }));

                                            // Update CSS variables for live preview
                                            const color = tinycolor(newColor);
                                            document.documentElement.style.setProperty('--color-primary-500', newColor);
                                            document.documentElement.style.setProperty('--color-primary-50', color.lighten(35).toHexString());
                                            document.documentElement.style.setProperty('--color-primary-600', color.darken(10).toHexString());
                                            document.documentElement.style.setProperty('--color-primary-700', color.darken(15).toHexString());
                                        }}
                                        className="w-full h-10 p-1 border border-gray-300 rounded-lg"
                                    />
                                </div>

                                <div className="bg-gray-50 rounded-lg p-4 col-span-1 md:col-span-2">
                                    <h4 className="font-medium text-gray-900 dark:text-gray-100 mb-2">Logo</h4>
                                    <input
                                        type="file"
                                        accept="image/png, image/jpeg, image/svg+xml"
                                        onChange={(e) => e.target.files && handleBrandingFileUpload('logo', e.target.files[0])}
                                        className="mb-2"
                                    />
                                    {brandingConfig.brand_logo_url && (
                                        <div className="mt-2 flex items-center gap-4">
                                            <img src={brandingConfig.brand_logo_url} alt={brandingConfig.brand_name || 'Logo Preview'} className="max-h-12 border p-1 rounded" />
                                            <button onClick={() => setBrandingConfig(c => ({ ...c, brand_logo_url: '' }))} className="text-red-500 hover:text-red-700 text-sm">Xóa</button>
                                        </div>
                                    )}
                                    <div className="mt-4">
                                        <label className="block text-sm font-medium text-gray-700">Chiều cao Logo: {brandingConfig.brand_logo_height}</label>
                                        <input
                                            type="range"
                                            min="20"
                                            max="80"
                                            step="1"
                                            value={parseInt(brandingConfig.brand_logo_height) || 32}
                                            onChange={(e) => setBrandingConfig(c => ({ ...c, brand_logo_height: `${e.target.value}px` }))}
                                            className="w-full mt-1"
                                        />
                                    </div>
                                </div>


                            </div>

                            <div className="flex justify-end">
                                <button className="btn-primary" onClick={saveBrandingConfig}>
                                    Lưu cấu hình Thương hiệu
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default AdminDashboard