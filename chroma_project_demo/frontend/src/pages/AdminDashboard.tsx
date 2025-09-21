import React, { useState, useEffect, useMemo } from 'react'
import { Users, MessageSquare, Settings, BarChart3, Palette, Crown, ThumbsDown, Image as ImageIcon, Search, Pencil, Trash2 } from 'lucide-react'
import { api } from '../services/api'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'

import { useBranding } from '../contexts/BrandingContext'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

interface User {
    id: string
    username: string
    email: string
    full_name: string
    is_admin: boolean
    is_active: boolean
    account_status?: string
    role?: string
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

interface IgnoredQuestion {
    id: number;
    question: string;
    ignored_at: string;
    ignored_by_username: string;
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
    const navigate = useNavigate()
    const [stats, setStats] = useState<ChatStats | null>(null)
    const [userPage, setUserPage] = useState(1)
    const [userTotal, setUserTotal] = useState(0)
    const USERS_PER_PAGE = 10
    const [userFilters, setUserFilters] = useState({ role: 'all', status: 'all' })

    const [faqs, setFaqs] = useState<FAQItem[]>([]);
    const [suggestedFaqs, setSuggestedFaqs] = useState<SuggestedFAQItem[]>([]);
    const [ignoredQuestions, setIgnoredQuestions] = useState<IgnoredQuestion[]>([]);


    // Super admin usernames that should display crown and cannot be modified
    const isSuperAdminUser = (u: User) => {
        const uname = (u?.username || '').toLowerCase();
        return uname === 'admin' || uname === 'sg0510';
    }


    const loadFaqs = async () => {
        try {
            const [faqsRes, ignoredQuestionsRes] = await Promise.all([
                api.get('/admin/faqs'),
                api.get('/admin/ignored-questions')
            ]);
            setFaqs(faqsRes.data ?? []);
            setIgnoredQuestions(ignoredQuestionsRes.data ?? []);
        } catch (error) {
            toast.error('Không thể tải dữ liệu FAQs');
        }
    };

    const [loading, setLoading] = useState(false)
    const [activity, setActivity] = useState<any[]>([]);
    const [tokenActivity, setTokenActivity] = useState<any[]>([]);
    const [frequentQuestions, setFrequentQuestions] = useState<{question: string, count: number, id: string}[]>([]);
    // const [showCreate, setShowCreate] = useState(false)
    // const [form, setForm] = useState({ question: '', answer: '', category: '' })
    const [isFaqModalOpen, setIsFaqModalOpen] = useState(false);
    const [editingFaq, setEditingFaq] = useState<FAQItem | null>(null);
    const [faqForm, setFaqForm] = useState({ id: '', question: '', answer: '', category: '' });

    const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
    const [faqToDelete, setFaqToDelete] = useState<string | null>(null);

    const [sourceSuggestionId, setSourceSuggestionId] = useState<string | null>(null);

    const getInitial = (s?: string) => (s && s.trim().length > 0 ? s.trim().charAt(0).toUpperCase() : '?');

    // Feedback filters/state
    const [feedbackSearch, setFeedbackSearch] = useState('');
    const [feedbackRange, setFeedbackRange] = useState<'7' | '30' | 'all'>('7');
    const [feedbackType, setFeedbackType] = useState<'negative' | 'all'>('negative');

    const filteredFeedback = useMemo(() => {
        let arr = [...feedbackChats];
        // filter by date range
        const now = new Date();
        if (feedbackRange !== 'all') {
            const days = feedbackRange === '7' ? 7 : 30;
            const threshold = new Date(now);
            threshold.setDate(threshold.getDate() - days);
            arr = arr.filter(x => new Date(x.timestamp) >= threshold);
        }
        // filter by text
        if (feedbackSearch.trim()) {
            const q = feedbackSearch.toLowerCase();
            arr = arr.filter(x => (x.user_question || '').toLowerCase().includes(q) || (x.assistant_response || '').toLowerCase().includes(q));
        }
        // feedbackType: current dataset is negative only; keep for extensibility
        return arr;
    }, [feedbackChats, feedbackRange, feedbackSearch, feedbackType]);

    const feedbackTrend = useMemo(() => {
        // group by day label
        const map: Record<string, number> = {};
        filteredFeedback.forEach(f => {
            const d = new Date(f.timestamp);
            const label = d.toLocaleDateString('vi-VN', { weekday: 'short' });
            map[label] = (map[label] || 0) + 1;
        });
        return Object.entries(map).map(([day, count]) => ({ day, count }));
    }, [filteredFeedback]);

    // Branding upload (drag & drop)
    const onLogoDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
    };
    const onLogoDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) handleBrandingFileUpload('logo', file);
    };

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

    const handleCreateFaqFromQuestion = (question: string) => {
        openFaqModal(null);
        setFaqForm({ id: '', question, answer: '', category: 'General' });
        setSourceSuggestionId(null);
    };

    const handleIgnoreFrequentQuestion = async (question: string) => {
        try {
            await api.post('/admin/frequent-questions/ignore', { question });
            setFrequentQuestions(frequentQuestions.filter(q => q.question !== question));
            toast.success('Đã ẩn câu hỏi');
        } catch (error) {
            toast.error('Không thể ẩn câu hỏi');
        }
    };

    const handleUnignoreQuestion = async (id: number) => {
        try {
            await api.delete(`/admin/ignored-questions/${id}`);
            setIgnoredQuestions(ignoredQuestions.filter(q => q.id !== id));
            toast.success('Đã khôi phục câu hỏi');
        } catch (error) {
            toast.error('Không thể khôi phục câu hỏi');
        }
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
                            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-6">
                                {[
                                    {
                                        icon: <MessageSquare className="h-6 w-6 text-emerald-600" />,
                                        label: 'Tổng phiên chat',
                                        value: stats.total_sessions.toLocaleString()
                                    },
                                    {
                                        icon: <Users className="h-6 w-6 text-emerald-600" />,
                                        label: 'User hoạt động (hôm nay)',
                                        value: stats.active_users_today.toLocaleString()
                                    },
                                    {
                                        icon: <Users className="h-6 w-6 text-emerald-600" />,
                                        label: 'User hoạt động (tuần)',
                                        value: stats.active_users_week.toLocaleString()
                                    },
                                    {
                                        icon: <BarChart3 className="h-6 w-6 text-emerald-600" />,
                                        label: 'Tổng tin nhắn',
                                        value: stats.total_messages.toLocaleString()
                                    },
                                    {
                                        icon: <Settings className="h-6 w-6 text-emerald-600" />,
                                        label: 'Tổng tokens',
                                        value: stats.total_tokens.toLocaleString()
                                    },
                                    {
                                        icon: <Settings className="h-6 w-6 text-emerald-600" />,
                                        label: 'Thời gian phản hồi TB',
                                        value: `${stats.avg_response_time.toFixed(2)}s`
                                    }
                                ].map((card, idx) => (
                                    <div key={idx} className="bg-white rounded-xl border shadow-sm p-5 dark:bg-[#0f0f0f] dark:border-gray-800 transition-transform duration-200 transform-gpu hover:shadow-lg hover:-translate-y-0.5 hover:scale-[1.01]">
                                        <div className="flex items-center gap-4">
                                            <div className="h-10 w-10 rounded-lg bg-emerald-50 dark:bg-emerald-900/20 flex items-center justify-center">
                                                {card.icon}
                                            </div>
                                            <div>
                                                <p className="text-sm text-gray-500 dark:text-gray-400">{card.label}</p>
                                                <p className="text-2xl font-semibold text-gray-900 dark:text-gray-100">{card.value}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Charts & Tables Container */}
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* Activity Chart - Area with Gradient */}
                                <div className="bg-white rounded-lg p-6 border dark:bg-[#0f0f0f] dark:border-gray-800">
                                    <h3 className="text-lg font-semibold mb-4">Hoạt động chat (7 ngày qua)</h3>
                                    <div className="h-64">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={activity.map((x) => ({ day: new Date(x.date).toLocaleDateString('vi-VN', { weekday: 'short' }), messages: x.count }))}>
                                                <defs>
                                                    <linearGradient id="colorMessages" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                                                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="day" />
                                                <YAxis allowDecimals={false} />
                                                <Tooltip contentStyle={{ backgroundColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#0f0f0f' : '#ffffff', borderColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#1f2937' : '#e5e7eb', color: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#e5e7eb' : '#111827' }} />
                                                <Area type="monotone" dataKey="messages" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorMessages)" />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Token Chart - Area with Gradient */}
                                <div className="bg-white rounded-lg p-6 border dark:bg-[#0f0f0f] dark:border-gray-800">
                                    <h3 className="text-lg font-semibold mb-4">Lượng Token Sử Dụng (7 ngày qua)</h3>
                                    <div className="h-64">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={tokenActivity.map((x) => ({ day: new Date(x.date).toLocaleDateString('vi-VN', { weekday: 'short' }), tokens: x.tokens }))}>
                                                <defs>
                                                    <linearGradient id="colorTokens" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                                                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" />
                                                <XAxis dataKey="day" />
                                                <YAxis allowDecimals={false} />
                                                <Tooltip contentStyle={{ backgroundColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#0f0f0f' : '#ffffff', borderColor: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#1f2937' : '#e5e7eb', color: (typeof document !== 'undefined' && document.documentElement.classList.contains('dark')) ? '#e5e7eb' : '#111827' }} />
                                                <Area type="monotone" dataKey="tokens" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#colorTokens)" />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>

                                {/* Frequent Questions Table */}
                                <div className="bg-white rounded-lg p-6 border dark:bg-[#0f0f0f] dark:border-gray-800">
                                    <h3 className="text-lg font-semibold mb-4">Câu hỏi thường gặp nhất</h3>
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
                                            <thead className="bg-gray-50 dark:bg-[#0f0f0f]">
                                                <tr>
                                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">Câu hỏi</th>
                                                    <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300 w-24">Số lần</th>
                                                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300 w-32">Hành động</th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200 dark:bg-[#0b0b0b] dark:divide-gray-800">
                                                {frequentQuestions.map((q, index) => (
                                                    <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                                                        <td className="px-4 py-3 whitespace-normal text-sm text-gray-900 dark:text-gray-100">{q.question}</td>
                                                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100 text-center">{q.count}</td>
                                                        <td className="px-4 py-3 whitespace-nowrap text-right text-sm space-x-2">
                                                            <button className="btn-primary text-xs px-2 py-1" onClick={() => handleCreateFaqFromQuestion(q.question)}>Tạo FAQ</button>
                                                            <button
                                                                onClick={() => handleIgnoreFrequentQuestion(q.question)}
                                                                className="p-1.5 rounded-md text-gray-500 hover:bg-red-100 hover:text-red-600 dark:text-gray-400 dark:hover:bg-red-900/20 dark:hover:text-red-400 transition-colors inline-flex items-center align-middle"
                                                                aria-label="Ẩn câu hỏi"
                                                                title="Ẩn câu hỏi"
                                                            >
                                                                <Trash2 className="h-4 w-4" />
                                                            </button>
                                                        </td>
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
                                <div className="flex items-center gap-4 whitespace-nowrap overflow-x-auto">
                                    <select
                                        value={userFilters.role}
                                        onChange={e => { setUserFilters(f => ({ ...f, role: e.target.value })); setUserPage(1); }}
                                        className="input-field text-sm py-1 w-44 md:w-56"
                                    >
                                        <option value="all">Mọi vai trò</option>
                                        <option value="admin">Admin</option>
                                        <option value="user">User</option>
                                    </select>
                                    <select
                                        value={userFilters.status}
                                        onChange={e => { setUserFilters(f => ({ ...f, status: e.target.value })); setUserPage(1); }}
                                        className="input-field text-sm py-1 w-44 md:w-56"
                                    >
                                        <option value="all">Mọi trạng thái</option>
                                        <option value="active">Hoạt động</option>
                                        <option value="inactive">Bị khóa</option>
                                    </select>
                                    <span className="badge badge-neutral">Tổng: {userTotal}</span>
                                </div>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                {users.map((user, index) => (
                                    <div key={user.id} className="bg-white rounded-xl border shadow-sm p-4 dark:bg-[#0f0f0f] dark:border-gray-800 hover:shadow-md transition-all">
                                        <div className="flex items-start gap-3">
                                            <div className="h-10 w-10 rounded-full flex items-center justify-center bg-gradient-to-br from-green-100 to-green-200 text-green-700 dark:from-green-900/30 dark:to-green-800/20 font-semibold">
                                                {(user.full_name || user.username || '?').charAt(0).toUpperCase()}
                                            </div>
                                            <div className="min-w-0">
                                                <div className="text-sm font-semibold text-gray-900 dark:text-gray-100 truncate">{user.full_name}</div>
                                                <div className="text-xs text-gray-500 dark:text-gray-400 truncate">@{user.username}</div>
                                            </div>
                                            <div className="ml-auto flex flex-wrap gap-1">
                                                <span className={`badge ${user.is_active ? 'badge-success' : 'badge-danger'}`}>{user.is_active ? 'Hoạt động' : 'Bị khóa'}</span>
                                                <span className={`badge ${user.is_admin ? 'badge-purple' : 'badge-neutral'}`}>
                                                    {user.is_admin ? (<>{isSuperAdminUser(user) && <Crown className="w-4 h-4 mr-1" />}{isSuperAdminUser(user) ? 'Owner' : 'Admin'}</>) : 'User'}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
                                            <div className="text-gray-600 dark:text-gray-300">
                                                <div className="text-xs text-gray-500 dark:text-gray-400">Email</div>
                                                <div className="truncate">{user.email}</div>
                                            </div>
                                            <div className="text-gray-600 dark:text-gray-300">
                                                <div className="text-xs text-gray-500 dark:text-gray-400">Chat count</div>
                                                <div>{user.chat_count}</div>
                                            </div>
                                        </div>
                                        <div className="mt-3 flex items-center justify-between">
                                            <button onClick={() => navigate(`/admin/users/${user.id}/edit`)} className="btn-ghost h-9">Sửa</button>
                                            <div className="flex items-center gap-4">
                                                <label className="relative inline-flex items-center cursor-pointer">
                                                    <input type="checkbox" checked={user.is_active} onChange={() => toggleUserStatus(user.id)} className="sr-only peer" disabled={isSuperAdminUser(user)} />
                                                    <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
                                                    <span className="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300 sr-only">User Status</span>
                                                </label>
                                                <label className="relative inline-flex items-center cursor-pointer">
                                                    <input type="checkbox" checked={user.is_admin} onChange={() => toggleAdminStatus(user.id)} className="sr-only peer" disabled={isSuperAdminUser(user)} />
                                                    <div className="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-purple-600"></div>
                                                    <span className="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300 sr-only">Admin Status</span>
                                                </label>
                                            </div>
                                        </div>
                                    </div>
                                ))}
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

                            {/* Existing FAQs */}
                            <div className="bg-white rounded-lg border p-6 dark:bg-[#0f0f0f] dark:border-gray-800">
                                <div className="flex justify-between items-center mb-4">
                                    <h3 className="text-lg font-semibold">Danh sách FAQs hiện tại</h3>
                                    <button className="btn-primary" onClick={() => openFaqModal(null)}>Thêm FAQ mới</button>
                                </div>
                                {faqs.length > 0 ? (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {faqs.map(faq => (
                                            <div key={faq.id} className="bg-white rounded-xl border shadow-sm p-4 dark:bg-[#0f0f0f] dark:border-gray-800 hover:shadow-md transition-transform transform-gpu hover:-translate-y-0.5 hover:scale-[1.01]">
                                                <div className="flex items-start gap-3">

                                                    <div className="h-10 w-10 rounded-full bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 flex items-center justify-center font-semibold">
                                                        {getInitial(faq.question)}
                                                    </div>
                                                    <div className="min-w-0">
                                                        <p className="font-medium text-gray-900 dark:text-gray-100 truncate">{faq.question}</p>
                                                        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mt-1">{faq.answer}</p>
                                                        <div className="mt-2">
                                                            <span className="badge badge-neutral">{faq.category || 'General'}</span>
                                                        </div>
                                                    </div>
                                                    <div className="ml-auto flex items-center gap-0.5">
                                                        <button
                                                            onClick={() => openFaqModal(faq)}
                                                            className="p-1.5 rounded-md text-gray-500 hover:bg-gray-100 hover:text-gray-800 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-200 transition-colors"
                                                            aria-label="Sửa FAQ"
                                                            title="Sửa FAQ"
                                                        >
                                                            <Pencil className="h-4 w-4" />
                                                        </button>
                                                        <button
                                                            onClick={() => openDeleteModal(faq.id)}
                                                            className="p-1.5 rounded-md text-gray-500 hover:bg-red-100 hover:text-red-600 dark:text-gray-400 dark:hover:bg-red-900/20 dark:hover:text-red-400 transition-colors"
                                                            aria-label="Xóa FAQ"
                                                            title="Xóa FAQ"
                                                        >
                                                            <Trash2 className="h-4 w-4" />
                                                        </button>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-sm text-gray-500">Chưa có FAQ nào.</p>
                                )}
                            </div>

                            {/* Ignored Frequent Questions */}
                            <div className="bg-white rounded-lg border p-6 dark:bg-[#0f0f0f] dark:border-gray-800">
                                <h3 className="text-lg font-semibold mb-4">Câu hỏi thường gặp đã ẩn</h3>
                                {ignoredQuestions.length > 0 ? (
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
                                            <thead className="bg-gray-50 dark:bg-[#0f0f0f]">
                                                <tr>
                                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">Câu hỏi</th>
                                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">Người ẩn</th>
                                                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">Ngày ẩn</th>
                                                    <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider dark:text-gray-300">Hành động</th>
                                                </tr>
                                            </thead>
                                            <tbody className="bg-white divide-y divide-gray-200 dark:bg-[#0b0b0b] dark:divide-gray-800">
                                                {ignoredQuestions.map((q) => (
                                                    <tr key={q.id} className="hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
                                                        <td className="px-4 py-3 whitespace-normal text-sm text-gray-900 dark:text-gray-100">{q.question}</td>
                                                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{q.ignored_by_username}</td>
                                                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">{new Date(q.ignored_at).toLocaleDateString('vi-VN')}</td>
                                                        <td className="px-4 py-3 whitespace-nowrap text-right">
                                                            <button className="btn-secondary text-xs px-2 py-1" onClick={() => handleUnignoreQuestion(q.id)}>Khôi phục</button>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                ) : (
                                    <p className="text-sm text-gray-500">Không có câu hỏi nào bị ẩn.</p>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Config Tab */}


                    {activeTab === 'config' && (
                        <div className="space-y-8">
                            <div>
                                <h3 className="text-xl font-bold">Cấu hình Hệ thống</h3>
                                <p className="text-sm text-gray-500">Tinh chỉnh các tham số cốt lõi của mô hình AI và hệ thống.</p>
                            </div>

                            <div className="bg-white rounded-xl border shadow-sm dark:bg-[#0f0f0f] dark:border-gray-800">
                                <div className="p-6 border-b dark:border-gray-800">
                                    <h4 className="text-lg font-semibold">Cấu hình AI</h4>
                                    <p className="text-sm text-gray-500">Chọn mô hình ngôn ngữ và các tham số liên quan đến hiệu suất tìm kiếm.</p>
                                </div>
                                <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                                    {/* Chat Model */}
                                    <div>
                                        <label className="font-medium text-gray-900 dark:text-gray-100">Mô hình AI</label>
                                        <p className="text-xs text-gray-500 mb-2">Mô hình xử lý và trả lời câu hỏi.</p>
                                        <select className="input-field" value={config.chat_model} onChange={(e) => setConfig((c) => ({ ...c, chat_model: e.target.value }))}>
                                            <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                                            <option value="gpt-4">GPT-4</option>
                                            <option value="gpt-4-turbo">GPT-4 Turbo</option>
                                            <option value="gpt-4o-mini">gpt-4o-mini</option>
                                        </select>
                                    </div>

                                    {/* Embedding Model */}
                                    <div>
                                        <label className="font-medium text-gray-900 dark:text-gray-100">Embedding Model</label>
                                        <p className="text-xs text-gray-500 mb-2">Mô hình mã hóa văn bản để tìm kiếm.</p>
                                        <select className="input-field" value={config.embed_model} onChange={(e) => setConfig((c) => ({ ...c, embed_model: e.target.value }))}>
                                            <option value="text-embedding-3-small">text-embedding-3-small</option>
                                            <option value="text-embedding-3-large">text-embedding-3-large</option>
                                        </select>
                                    </div>

                                    {/* Similarity Threshold */}
                                    <div className="md:col-span-1">
                                        <label className="font-medium text-gray-900 dark:text-gray-100">Ngưỡng Similarity</label>
                                        <p className="text-xs text-gray-500 mb-2">Mức độ tương đồng tối thiểu để coi là liên quan.</p>
                                        <input
                                            type="range" min="0" max="1" step="0.05"
                                            value={config.similarity_threshold}
                                            onChange={(e) => setConfig((c) => ({ ...c, similarity_threshold: parseFloat(e.target.value) }))}
                                            className="w-full accent-emerald-600"
                                        />
                                        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                                            <span>0.0</span>
                                            <span className="font-semibold text-emerald-600">{config.similarity_threshold.toFixed(2)}</span>
                                            <span>1.0</span>
                                        </div>
                                    </div>

                                    {/* Max Tokens */}
                                    <div>
                                        <label className="font-medium text-gray-900 dark:text-gray-100">Max Tokens</label>
                                        <p className="text-xs text-gray-500 mb-2">Số token tối đa trong một câu trả lời.</p>
                                        <input
                                            type="number" value={config.max_tokens}
                                            onChange={(e) => setConfig((c) => ({ ...c, max_tokens: parseInt(e.target.value || '0') }))}
                                            className="input-field"
                                            min="100" max="128000"
                                        />
                                    </div>
                                </div>
                                <div className="p-4 bg-gray-50 dark:bg-zinc-800/50 rounded-b-xl flex justify-end">
                                    <button className="btn-primary" onClick={saveConfigs}>Lưu cấu hình AI</button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Branding Tab */}
                    {activeTab === 'branding' && (
                        <div className="space-y-8">
                            <div>
                                <h3 className="text-xl font-bold">Tùy chỉnh Thương hiệu</h3>
                                <p className="text-sm text-gray-500">Cá nhân hóa giao diện của chatbot để phù hợp với thương hiệu của bạn.</p>
                            </div>

                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                                {/* Left Side - Configs */}
                                <div className="lg:col-span-2 space-y-6">
                                    {/* Brand Name & Color */}
                                    <div className="bg-white rounded-xl border shadow-sm dark:bg-[#0f0f0f] dark:border-gray-800">
                                        <div className="p-6">
                                            <h4 className="text-lg font-semibold">Thông tin cơ bản</h4>
                                            <p className="text-sm text-gray-500">Đặt tên và màu sắc chủ đạo cho chatbot.</p>
                                        </div>
                                        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-6 border-t dark:border-gray-800">
                                            <div>
                                                <label className="font-medium text-gray-900 dark:text-gray-100">Tên thương hiệu</label>
                                                <input
                                                    type="text"
                                                    value={brandingConfig.brand_name}
                                                    onChange={(e) => setBrandingConfig(c => ({ ...c, brand_name: e.target.value }))}
                                                    className="input-field mt-2"
                                                    placeholder="e.g., Chroma AI Chat"
                                                />
                                            </div>
                                            <div>
                                                <label className="font-medium text-gray-900 dark:text-gray-100">Màu chủ đề</label>
                                                <div className="relative mt-2">
                                                    <input
                                                        type="text"
                                                        value={brandingConfig.primary_color}
                                                        onChange={(e) => setBrandingConfig(c => ({ ...c, primary_color: e.target.value }))}
                                                        className="input-field w-full pr-12"
                                                    />
                                                    <input
                                                        type="color"
                                                        value={brandingConfig.primary_color}
                                                        onChange={(e) => setBrandingConfig(c => ({ ...c, primary_color: e.target.value }))}
                                                        className="absolute right-1 top-1/2 -translate-y-1/2 w-8 h-8 p-0 border-none rounded-md cursor-pointer bg-transparent"
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Logo Upload */}
                                    <div className="bg-white rounded-xl border shadow-sm dark:bg-[#0f0f0f] dark:border-gray-800">
                                        <div className="p-6">
                                            <h4 className="text-lg font-semibold">Logo Thương hiệu</h4>
                                            <p className="text-sm text-gray-500">Tải lên logo để hiển thị trên giao diện chat.</p>
                                        </div>
                                        <div className="p-6 border-t dark:border-gray-800">
                                            <div
                                                onDragOver={onLogoDragOver}
                                                onDrop={onLogoDrop}
                                                className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-emerald-500 dark:hover:border-emerald-600 transition-colors"
                                            >
                                                <input
                                                    type="file" id="logo-upload" className="hidden"
                                                    accept="image/png, image/jpeg, image/svg+xml"
                                                    onChange={(e) => e.target.files && handleBrandingFileUpload('logo', e.target.files[0])}
                                                />
                                                <label htmlFor="logo-upload" className="cursor-pointer">
                                                    <ImageIcon className="mx-auto h-10 w-10 text-gray-400" />
                                                    <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
                                                        <span className="font-semibold text-emerald-600">Nhấn để tải lên</span> hoặc kéo thả
                                                    </p>
                                                    <p className="text-xs text-gray-500">PNG, JPG, SVG (tối đa 2MB)</p>
                                                </label>
                                            </div>
                                            {brandingConfig.brand_logo_url && (
                                                <div className="mt-4">
                                                    <label className="font-medium text-gray-900 dark:text-gray-100">Chiều cao Logo: {parseInt(brandingConfig.brand_logo_height)}px</label>
                                                    <input
                                                        type="range" min="20" max="80" step="1"
                                                        value={parseInt(brandingConfig.brand_logo_height) || 32}
                                                        onChange={(e) => setBrandingConfig(c => ({ ...c, brand_logo_height: `${e.target.value}px` }))}
                                                        className="w-full mt-2 accent-emerald-600"
                                                    />
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* Right Side - Preview */}
                                <div className="lg:col-span-1">
                                    <div className="sticky top-24">
                                        <h4 className="font-semibold mb-2">Xem trước</h4>
                                        <div className="rounded-xl border shadow-sm w-full bg-white dark:bg-[#0f0f0f] dark:border-gray-800 p-4">
                                            <div className="flex items-center gap-3">
                                                {brandingConfig.brand_logo_url ? (
                                                    <img src={brandingConfig.brand_logo_url} alt="Logo Preview" style={{ height: brandingConfig.brand_logo_height }} />
                                                ) : (
                                                    <div className="w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                                                        <ImageIcon className="w-6 h-6 text-gray-500" />
                                                    </div>
                                                )}
                                                <h3 className="font-bold text-lg" style={{ color: brandingConfig.primary_color }}>
                                                    {brandingConfig.brand_name || 'Tên Thương hiệu'}
                                                </h3>
                                            </div>
                                            <div className="mt-4 space-y-2">
                                                <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-sm w-3/4">Đây là tin nhắn của người dùng.</div>
                                                <div className="flex justify-end">
                                                    <div className="p-2 rounded-lg text-white text-sm w-3/4" style={{ backgroundColor: brandingConfig.primary_color }}>
                                                        Đây là câu trả lời từ AI, với màu sắc thương hiệu của bạn.
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex justify-end mt-8">
                                <button className="btn-primary" onClick={saveBrandingConfig}>
                                    Lưu cấu hình Thương hiệu
                                </button>
                            </div>
                        </div>
                    )}


                        {activeTab === 'feedback' && (
                            <div className="space-y-6">
                                {/* Filters and Header */}
                                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                                    <div>
                                        <h3 className="text-xl font-bold">Phân tích Phản hồi</h3>
                                        <p className="text-sm text-gray-500">Xem lại các tương tác bị người dùng đánh giá thấp.</p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <div className="relative">
                                            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400" />
                                            <input
                                                type="text"
                                                placeholder="Tìm kiếm câu hỏi..."
                                                value={feedbackSearch}
                                                onChange={(e) => setFeedbackSearch(e.target.value)}
                                                className="input-field pl-9 w-40 sm:w-auto"
                                            />
                                        </div>
                                        <select value={feedbackRange} onChange={e => setFeedbackRange(e.target.value as any)} className="input-field">
                                            <option value="7">7 ngày qua</option>
                                            <option value="30">30 ngày qua</option>
                                            <option value="all">Tất cả</option>
                                        </select>
                                    </div>
                                </div>

                                {/* Chart and Stats */}
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                    <div className="lg:col-span-2 bg-white rounded-xl border shadow-sm p-5 dark:bg-[#0f0f0f] dark:border-gray-800">
                                        <h4 className="font-semibold mb-4">Xu hướng Phản hồi tiêu cực</h4>
                                        <ResponsiveContainer width="100%" height={200}>
                                            <BarChart data={feedbackTrend} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                                                <XAxis dataKey="day" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                                                <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
                                                <Tooltip
                                                    contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '0.5rem' }}
                                                    labelStyle={{ color: '#d1d5db' }}
                                                    itemStyle={{ color: '#9ca3af' }}
                                                />
                                                <Bar dataKey="count" fill="#ef4444" radius={[4, 4, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                    <div className="bg-white rounded-xl border shadow-sm p-5 dark:bg-[#0f0f0f] dark:border-gray-800 flex flex-col justify-center items-center text-center">
                                        <h4 className="font-semibold text-gray-600 dark:text-gray-400">Tổng Phản hồi tiêu cực</h4>
                                        <p className="text-5xl font-bold text-red-600 dark:text-red-500 mt-2">{filteredFeedback.length}</p>
                                        <p className="text-xs text-gray-500 mt-1">trong khoảng thời gian đã chọn</p>
                                    </div>
                                </div>

                                {/* Feedback List */}
                                <div className="space-y-4">
                                    {filteredFeedback.length > 0 ? (
                                        filteredFeedback.map(chat => (
                                            <div key={chat.feedback_id} className="bg-white rounded-xl border shadow-sm p-4 dark:bg-[#0f0f0f] dark:border-gray-800 transition-shadow hover:shadow-md">
                                                <div className="flex justify-between items-start">
                                                    <div>
                                                        <p className="text-xs text-gray-500 dark:text-gray-400">{new Date(chat.timestamp).toLocaleString('vi-VN')}</p>
                                                        <p className="mt-2 font-medium text-gray-800 dark:text-gray-200">
                                                            <span className="font-bold">Hỏi:</span> {chat.user_question}
                                                        </p>
                                                        <div className="mt-1 text-sm text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/20 p-2 rounded-md">
                                                            <p><span className="font-bold">Đáp (Bị đánh giá thấp):</span> {chat.assistant_response}</p>
                                                        </div>
                                                    </div>
                                                    <a href={`/dashboard?sid=${chat.session_id}`} target="_blank" rel="noopener noreferrer" className="btn-secondary text-xs px-2 py-1 ml-4 flex-shrink-0">
                                                        Xem chi tiết
                                                    </a>
                                                </div>
                                            </div>
                                        ))
                                    ) : (
                                        <div className="text-center py-10">
                                            <p className="text-gray-500">Không có phản hồi tiêu cực nào khớp với bộ lọc của bạn.</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

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


