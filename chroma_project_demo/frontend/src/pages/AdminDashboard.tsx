import React, { useState, useEffect } from 'react'
import { Users, MessageSquare, Settings, BarChart3, Palette, Crown, ThumbsDown } from 'lucide-react'
import { api } from '../services/api'
import toast from 'react-hot-toast'

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


interface FAQItem {
    id: string;
    question: string;
    answer: string;
    category: string;
}

interface SuggestedFAQItem {
    id: string;
    question: string;
    source_count: number;
}


interface FeedbackChat {
    feedback_id: number;
    chat_message_id: string;
    session_id: string;
    user_question: string;
    assistant_response: string;
    timestamp: string;
}

type TabId = 'overview' | 'users' | 'config' | 'branding' | 'faqs' | 'feedback'

interface AdminDashboardProps {
    initialTab?: TabId
}

const AdminDashboard: React.FC<AdminDashboardProps> = ({ initialTab = 'overview' }) => {
    const [activeTab, setActiveTab] = useState<TabId>(initialTab)
    const { brandingConfig: contextBrandingConfig, loadBrandingConfig: reloadBranding } = useBranding()
    const [feedbackChats, setFeedbackChats] = useState<FeedbackChat[]>([]);
    const [users, setUsers] = useState<User[]>([])
    const [stats, setStats] = useState<ChatStats | null>(null)
    const [userPage, setUserPage] = useState(1)
    const [userTotal, setUserTotal] = useState(0)
    const USERS_PER_PAGE = 10
    const [userFilters, setUserFilters] = useState({ role: 'all', status: 'all' })

    const [faqs, setFaqs] = useState<FAQItem[]>([]);
    const [suggestedFaqs, setSuggestedFaqs] = useState<SuggestedFAQItem[]>([]);



    const loadFaqs = async () => {
        try {
            const [faqsRes, suggestedFaqsRes] = await Promise.all([
                api.get('/admin/faqs'),
                api.get('/admin/suggested-faqs')
            ]);
            setFaqs(faqsRes.data ?? []);

            setSuggestedFaqs(suggestedFaqsRes.data ?? []);
        } catch (error) {
            toast.error('Không thể tải dữ liệu FAQs');
        }
    };

    const [loading, setLoading] = useState(false)
    const [activity, setActivity] = useState<any[]>([]);
    const [tokenActivity, setTokenActivity] = useState<any[]>([]);
    const [frequentQuestions, setFrequentQuestions] = useState<{question: string, count: number}[]>([]);
    // const [showCreate, setShowCreate] = useState(false)
    // const [form, setForm] = useState({ question: '', answer: '', category: '' })
    const [isFaqModalOpen, setIsFaqModalOpen] = useState(false);
    const [editingFaq, setEditingFaq] = useState<FAQItem | null>(null);
    const [faqForm, setFaqForm] = useState({ id: '', question: '', answer: '', category: '' });

    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [faqToDelete, setFaqToDelete] = useState<string | null>(null);

    const [sourceSuggestionId, setSourceSuggestionId] = useState<string | null>(null);
    const openFaqModal = (faq: FAQItem | null) => {
        if (faq) {

            setEditingFaq(faq);


            setFaqForm(faq);
        } else {
            setEditingFaq(null);
            setFaqForm({ id: '', question: '', answer: '', category: 'General' });
        }
        setIsFaqModalOpen(true);
    };

    const closeFaqModal = () => {
        setIsFaqModalOpen(false);
        setEditingFaq(null);
        setFaqForm({ id: '', question: '', answer: '', category: '' });
    };

    const handleFaqFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        const { name, value } = e.target;
        setFaqForm(prev => ({ ...prev, [name]: value }));
    };

    const handleFaqSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            let newFaq: FAQItem;
            if (editingFaq) {
                const res = await api.put(`/admin/faqs/${faqForm.id}`, faqForm);
                newFaq = res.data;
                setFaqs(faqs.map(f => f.id === newFaq.id ? newFaq : f));
                toast.success('FAQ đã được cập nhật!');
            } else {
                const res = await api.post('/admin/faqs', {
                    question: faqForm.question,
                    answer: faqForm.answer,
                    category: faqForm.category
                });
                newFaq = res.data;
                setFaqs([...faqs, newFaq]);
                toast.success('FAQ đã được thêm mới!');

                if (sourceSuggestionId) {
                    await api.delete(`/admin/suggested-faqs/${sourceSuggestionId}`);
                    setSuggestedFaqs(suggestedFaqs.filter(s => s.id !== sourceSuggestionId));
                }
            }
            closeFaqModal();
        } catch (error) {
            toast.error(`Không thể ${editingFaq ? 'cập nhật' : 'thêm mới'} FAQ`);
        }
    };

    const openDeleteModal = (id: string) => {
        setFaqToDelete(id);
        setIsDeleteModalOpen(true);
    };

    const closeDeleteModal = () => {
        setFaqToDelete(null);
        setIsDeleteModalOpen(false);
    };

    const handleDeleteFaq = async () => {
        if (faqToDelete) {
            try {
                await api.delete(`/admin/faqs/${faqToDelete}`);
                setFaqs(faqs.filter(f => f.id !== faqToDelete));
                toast.success('FAQ đã được xóa!');
                closeDeleteModal();
            } catch (error) {
                toast.error('Không thể xóa FAQ');
            }
        }
    };

    const handleRejectSuggestedFaq = async (id: string) => {
        try {
            await api.delete(`/admin/suggested-faqs/${id}`);
            setSuggestedFaqs(suggestedFaqs.filter(s => s.id !== id));
            toast.success('Đã từ chối đề xuất');
        } catch (error) {
            toast.error('Không thể từ chối đề xuất');
        }
    };

    const handleAcceptSuggestedFaq = (faq: SuggestedFAQItem) => {
        openFaqModal(null);
        setFaqForm({ id: '', question: faq.question, answer: '', category: 'General' });
        setSourceSuggestionId(faq.id);
    };

    const loadFeedbackChats = async () => {
        try {
            const res = await api.get('/admin/feedback-chats');
            setFeedbackChats(res.data ?? []);
        } catch (error) {
            toast.error('Không thể tải dữ liệu phản hồi');
        }
    };

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
            const configsToSave = [
                { key: 'chat_model', value: String(config.chat_model), description: 'chat model' },
                { key: 'embed_model', value: String(config.embed_model), description: 'embed model' },
                { key: 'similarity_threshold', value: String(config.similarity_threshold), description: 'similarity threshold' },
                { key: 'max_tokens', value: String(config.max_tokens), description: 'max tokens' },
            ];
            await api.put('/admin/system-configs', configsToSave);
            toast.success('Đã lưu cấu hình');
        } catch (e: any) {
            toast.error(e?.response?.data?.detail || 'Lưu cấu hình thất bại');
        }
    };

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

    const loadUsers = async () => {
        try {
            const params = new URLSearchParams({
                page: String(userPage),
                limit: String(USERS_PER_PAGE),
                role: userFilters.role,
                status: userFilters.status
            });
            const res = await api.get(`/admin/users?${params.toString()}`);
            setUsers(res.data.users ?? []);
            setUserTotal(res.data.total ?? 0);
        } catch (error: any) {
            toast.error(`Không thể tải danh sách user: ${error?.response?.status || error?.message}`);
        }
    };

    useEffect(() => {
        const loadOtherData = async () => {
            setLoading(true)
            try {
                const [statsRes, activityRes, tokenActivityRes, frequentQuestionsRes] = await Promise.all([
                    api.get('/admin/stats'),
                    api.get('/admin/activity'),
                    api.get('/admin/token-activity'),
                    api.get('/admin/frequent-questions')
                ])
                setStats(statsRes.data ?? null)
                setActivity(activityRes.data ?? [])
                setTokenActivity(tokenActivityRes.data ?? [])
                setFrequentQuestions(frequentQuestionsRes.data ?? [])
                await loadConfigs()
            } catch (error: any) {
                toast.error(`Không thể tải dữ liệu dashboard: ${error?.response?.status || error?.message}`)
            } finally {
                setLoading(false)
            }
        }
        loadOtherData()
    }, [])

    useEffect(() => {
        if (activeTab === 'users') {
            loadUsers()
        } else if (activeTab === 'faqs') {
            loadFaqs()
        } else if (activeTab === 'feedback') {
            loadFeedbackChats()
        }
    }, [userPage, activeTab, userFilters])

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
        { id: 'faqs', name: 'Quản lý FAQs', icon: MessageSquare },
        { id: 'feedback', name: 'Phân tích Phản hồi', icon: ThumbsDown },
        { id: 'config', name: 'Cấu hình', icon: Settings },

        { id: 'branding', name: 'Thương hiệu', icon: Palette },
    ]

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
            </div>
        );
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
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <MessageSquare className="h-8 w-8 text-blue-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-blue-600 dark:text-blue-400">Tổng phiên chat</p>
                                            <p className="text-2xl font-semibold text-blue-900 dark:text-blue-200">{stats.total_sessions}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-green-50 dark:bg-green-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <Users className="h-8 w-8 text-green-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-green-600 dark:text-green-400">User hoạt động (hôm nay)</p>
                                            <p className="text-2xl font-semibold text-green-900 dark:text-green-200">{stats.active_users_today}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-yellow-50 dark:bg-yellow-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <Users className="h-8 w-8 text-yellow-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-yellow-600 dark:text-yellow-400">User hoạt động (tuần)</p>
                                            <p className="text-2xl font-semibold text-yellow-900 dark:text-yellow-200">{stats.active_users_week}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-indigo-50 dark:bg-indigo-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <BarChart3 className="h-8 w-8 text-indigo-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400">Tổng tin nhắn</p>
                                            <p className="text-2xl font-semibold text-indigo-900 dark:text-indigo-200">{stats.total_messages.toLocaleString()}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-pink-50 dark:bg-pink-950/30 rounded-lg p-6">
                                    <div className="flex items-center">
                                        <div className="flex-shrink-0">
                                            <Settings className="h-8 w-8 text-pink-600" />
                                        </div>
                                        <div className="ml-4">
                                            <p className="text-sm font-medium text-pink-600 dark:text-pink-400">Tổng tokens</p>
                                            <p className="text-2xl font-semibold text-pink-900 dark:text-pink-200">{stats.total_tokens.toLocaleString()}</p>
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

                            {/* Charts & Tables Container */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Activity Chart */}
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

                                {/* Token Chart */}
                                <div className="bg-white rounded-lg p-6 border dark:bg-[#0f0f0f] dark:border-gray-800">
                                    <h3 className="text-lg font-semibold mb-4">Lượng Token Sử Dụng (7 ngày qua)</h3>
                                    <div className="h-64">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={tokenActivity.map((x) => ({ day: new Date(x.date).toLocaleDateString('vi-VN', { weekday: 'short' }), tokens: x.tokens }))}>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="day" />
                                                <YAxis />
                                                <Tooltip contentStyle={{ backgroundColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#0f0f0f' : '#ffffff', borderColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#1f2937' : '#e5e7eb', color: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#e5e7eb' : '#111827' }} />
                                                <Line type="monotone" dataKey="tokens" stroke="#8884d8" strokeWidth={2} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Frequent Questions Table */}
                                <div className="bg-white rounded-lg p-6 border dark:bg-[#0f0f0f] dark:border-gray-800 lg:col-span-2">
                                    <h3 className="text-lg font-semibold mb-4">Câu hỏi thường gặp nhất</h3>
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
                                            <thead className="bg-gray-50 dark:bg-[#0f0f0f]">
                                                <tr>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">Câu hỏi</th>
                                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300 w-24">Số lần hỏi</th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200 dark:bg-[#0b0b0b] dark:divide-gray-800">
                                                {frequentQuestions.map((q, index) => (
                                                    <tr key={index}>
                                                        <td className="px-6 py-4 whitespace-normal text-sm text-gray-900 dark:text-gray-100">{q.question}</td>
                                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 text-center">{q.count}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </div>

	                            </div>

                    )}

                    {/* Users Tab */}
                    {activeTab === 'users' && (
                        <div className="space-y-4">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-lg font-semibold">Danh sách User</h3>
                                <div className="flex items-center gap-4">
                                    <select
                                        value={userFilters.role}
                                        onChange={e => { setUserFilters(f => ({ ...f, role: e.target.value })); setUserPage(1); }}
                                        className="input-field text-sm py-1"
                                    >
                                        <option value="all">Mọi vai trò</option>
                                        <option value="admin">Admin</option>
                                        <option value="user">User</option>
                                    </select>
                                    <select
                                        value={userFilters.status}
                                        onChange={e => { setUserFilters(f => ({ ...f, status: e.target.value })); setUserPage(1); }}
                                        className="input-field text-sm py-1"
                                    >
                                        <option value="all">Mọi trạng thái</option>
                                        <option value="active">Hoạt động</option>
                                        <option value="inactive">Bị khóa</option>
                                    </select>
                                    <span className="text-sm text-gray-500 dark:text-gray-400">Tổng: {userTotal}</span>
                                </div>
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
                                                    <span className={`inline-flex items-center px-2 py-1 text-xs font-semibold rounded-full ${user.is_admin
                                                        ? 'bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300'
                                                        : 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200'
                                                        }`}>
                                                        {user.is_admin ? (
                                                            <>
                                                                {user.username === 'admin' && <Crown className="w-4 h-4 mr-1" />}
                                                                Admin
                                                            </>
                                                        ) : 'User'}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-4">
                                                    <label className="relative inline-flex items-center cursor-pointer">
                                                        <input type="checkbox" checked={user.is_active} onChange={() => toggleUserStatus(user.id)} className="sr-only peer" disabled={user.username === 'admin'} />
                                                        <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                                                        <span className="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300 sr-only">User Status</span>
                                                    </label>
                                                    <label className="relative inline-flex items-center cursor-pointer">
                                                        <input type="checkbox" checked={user.is_admin} onChange={() => toggleAdminStatus(user.id)} className="sr-only peer" disabled={user.username === 'admin'} />
                                                        <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-purple-600"></div>
                                                        <span className="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300 sr-only">Admin Status</span>
                                                    </label>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>

                            {/* Pagination */}
                            <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-800">
                                <span className="text-sm text-gray-600 dark:text-gray-400">
                                    Hiển thị {(userPage - 1) * USERS_PER_PAGE + 1} - {Math.min(userPage * USERS_PER_PAGE, userTotal)} trên {userTotal}
                                </span>
                                <div className="space-x-2">
                                    <button
                                        onClick={() => setUserPage(p => p - 1)}
                                        disabled={userPage === 1}
                                        className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Trang trước
                                    </button>
                                    <button
                                        onClick={() => setUserPage(p => p + 1)}
                                        disabled={userPage * USERS_PER_PAGE >= userTotal}
                                        className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        Trang sau
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* FAQs Tab */}
                    {activeTab === 'faqs' && (
                        <div className="space-y-6">
                            {/* Suggested FAQs */}
                            <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
                                <h3 className="text-lg font-semibold mb-4">Đề xuất câu hỏi mới</h3>
                                {suggestedFaqs.length > 0 ? (
                                    <ul className="space-y-3">
                                        {suggestedFaqs.map(sug => (
                                            <li key={sug.id} className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-md flex justify-between items-center">
                                                <div>
                                                    <p className="font-medium text-gray-800 dark:text-gray-200">{sug.question}</p>
                                                    <p className="text-xs text-gray-500">Số lần hỏi: {sug.source_count}</p>
                                                </div>
                                                <div className="space-x-2">
                                                    <button onClick={() => handleRejectSuggestedFaq(sug.id)} className="btn-secondary text-xs px-2 py-1">Từ chối</button>
                                                    <button onClick={() => handleAcceptSuggestedFaq(sug)} className="btn-primary text-xs px-2 py-1">Chấp nhận & Soạn thảo</button>
                                                </div>
                                            </li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p className="text-sm text-gray-500">Không có đề xuất nào.</p>
                                )}
                            </div>

                            {/* Existing FAQs */}
                            <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg font-semibold">Danh sách FAQs hiện tại</h3>
                                    <button className="btn-primary" onClick={() => openFaqModal(null)}>Thêm FAQ mới</button>
                                </div>
                                {faqs.length > 0 ? (
                                    <ul className="space-y-3">
                                        {faqs.map(faq => (
                                            <li key={faq.id} className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-md">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <p className="font-medium text-gray-800 dark:text-gray-200">{faq.question}</p>
                                                        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{faq.answer}</p>
                                                        <p className="text-xs text-gray-500 mt-2">Danh mục: {faq.category}</p>
                                                    </div>
                                                    <div className="space-x-2 flex-shrink-0 ml-4">
                                                        <button className="btn-secondary text-xs px-2 py-1" onClick={() => openFaqModal(faq)}>Sửa</button>
                                                        <button className="btn-danger text-xs px-2 py-1" onClick={() => openDeleteModal(faq.id)}>Xóa</button>
                                                    </div>
                                                </div>
                                            </li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p className="text-sm text-gray-500">Chưa có FAQ nào.</p>
                                )}
                            </div>
                        </div>
                    )}

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

                                            // Update only title color for live preview
                                            document.documentElement.style.setProperty('--brand-title-color', newColor);
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


                        {activeTab === 'feedback' && (
                            <div className="space-y-6">
                                <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
                                    <h3 className="text-lg font-semibold mb-4">Phản hồi tiêu cực từ người dùng</h3>
                                    {feedbackChats.length > 0 ? (
                                        <ul className="space-y-4">
                                            {feedbackChats.map(chat => (
                                                <li key={chat.feedback_id} className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-md border dark:border-gray-700">
                                                    <div className="flex justify-between items-start">
                                                        <div>
                                                            <p className="text-xs text-gray-500 dark:text-gray-400">{new Date(chat.timestamp).toLocaleString('vi-VN')}</p>
                                                            <p className="mt-2 font-medium text-gray-800 dark:text-gray-200">
                                                                <span className="font-bold">Hỏi:</span> {chat.user_question}
                                                            </p>
                                                            <p className="mt-1 text-sm text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/20 p-2 rounded-md">
                                                                <span className="font-bold">Đáp (Bị đánh giá thấp):</span> {chat.assistant_response}
                                                            </p>
                                                        </div>
                                                        <a href={`/dashboard?sid=${chat.session_id}`} target="_blank" rel="noopener noreferrer" className="btn-secondary text-xs px-2 py-1 ml-4 flex-shrink-0">
                                                            Xem chi tiết
                                                        </a>
                                                    </div>
                                                </li>
                                            ))}
                                        </ul>
                                    ) : (
                                        <p className="text-sm text-gray-500">Không có phản hồi tiêu cực nào.</p>
                                    )}
                                </div>
                            </div>
                        )}

                    {/* Feedback Tab */}

                </div>
            </div>

            {/* FAQ Modal */}
            {isFaqModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
                    <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-xl w-full max-w-2xl mx-4">
                        <form onSubmit={handleFaqSubmit}>
                            <div className="p-6 border-b dark:border-zinc-800">
                                <h3 className="text-lg font-semibold">{editingFaq ? 'Chỉnh sửa FAQ' : 'Thêm FAQ mới'}</h3>
                            </div>
                            <div className="p-6 space-y-4">
                                <div>
                                    <label htmlFor="question" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Câu hỏi</label>
                                    <input
                                        type="text"
                                        id="question"
                                        name="question"
                                        value={faqForm.question}
                                        onChange={handleFaqFormChange}
                                        className="input-field mt-1"
                                        required
                                    />
                                </div>
                                <div>
                                    <label htmlFor="answer" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Câu trả lời</label>
                                    <textarea
                                        id="answer"
                                        name="answer"
                                        value={faqForm.answer}
                                        onChange={handleFaqFormChange}
                                        rows={4}
                                        className="input-field mt-1"
                                        required
                                    />
                                </div>


                                <div>
                                    <label htmlFor="category" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Danh mục</label>
                                    <input
                                        type="text"
                                        id="category"
                                        name="category"
                                        value={faqForm.category}
                                        onChange={handleFaqFormChange}
                                        className="input-field mt-1"
                                        required
                                    />
                                </div>
                            </div>
                            <div className="p-6 flex justify-end gap-3 bg-gray-50 dark:bg-zinc-800/50 rounded-b-lg">
                                <button type="button" onClick={closeFaqModal} className="btn-secondary">Hủy</button>
                                <button type="submit" className="btn-primary">{editingFaq ? 'Lưu thay đổi' : 'Thêm FAQ'}</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
            {/* Delete Confirmation Modal */}
            {isDeleteModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
                    <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-xl w-full max-w-md mx-4">
                        <div className="p-6">
                            <h3 className="text-lg font-semibold">Xác nhận xóa</h3>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">Bạn có chắc chắn muốn xóa FAQ này không? Hành động này không thể hoàn tác.</p>
                        </div>
                        <div className="p-4 flex justify-end gap-3 bg-gray-50 dark:bg-zinc-800/50 rounded-b-lg">
                            <button type="button" onClick={closeDeleteModal} className="btn-secondary">Hủy</button>
                            <button type="button" onClick={handleDeleteFaq} className="btn-danger">Xóa</button>
                        </div>
                    </div>
                </div>
            )}


        </div>
    )



}

export default AdminDashboard


