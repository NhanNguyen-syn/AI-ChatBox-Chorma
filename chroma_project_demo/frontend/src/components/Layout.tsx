import React, { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useBranding } from '../contexts/BrandingContext'
import { useChatStore } from '../stores/chatStore'
import ThemeSwitch from './ThemeSwitch'
import {
    MessageSquare,
    Users,
    BarChart3,
    FileText,
    LogOut,
    User,
    MoreVertical,
    Trash2,
    AlertTriangle,
    Menu,
    X
} from 'lucide-react'

// Minimal chat sessions list for user sidebar with hover menu "..."
const ChatSessionsList: React.FC = () => {
    const { sessions, fetchSessions, deleteSession } = useChatStore()
    const [menuOpenId, setMenuOpenId] = useState<string | null>(null)
    const [confirmOpen, setConfirmOpen] = useState(false)
    const [pendingId, setPendingId] = useState<string | null>(null)
    const [deleting, setDeleting] = useState(false)

    const navigate = useNavigate()

    useEffect(() => {
        fetchSessions()

        const handleRefresh = () => fetchSessions()
        window.addEventListener('chat:sessions:refresh', handleRefresh)

        return () => {
            window.removeEventListener('chat:sessions:refresh', handleRefresh)
        }
    }, [fetchSessions])

    const openConfirm = (id: string) => {
        setPendingId(id)
        setConfirmOpen(true)
        setMenuOpenId(null)
    }
    const closeConfirm = () => { if (!deleting) { setConfirmOpen(false); setPendingId(null) } }

    const handleDelete = async (sessionId: string) => {
        try {
            setDeleting(true)
            await deleteSession(sessionId)

            // If current URL points to this sid, navigate to new chat
            const params = new URLSearchParams(window.location.search)
            if (params.get('sid') === sessionId) {
                navigate('/dashboard', { replace: true })
            }
        } finally {
            setDeleting(false)
            setConfirmOpen(false)
            setPendingId(null)
        }
    }

    return (
        <div className="space-y-1 pr-2">
            {sessions.length === 0 ? (
                <div className="text-xs text-gray-500 px-3">Chưa có đoạn chat</div>
            ) : (
                <ul className="space-y-1">
                    {sessions.map((s) => (
                        <li key={s.id} className="group relative">
                            <Link
                                to={`/dashboard?sid=${s.id}`}
                                className="block px-3 py-2 text-sm rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-200 truncate pr-8"
                                title={s.title || 'Đoạn Chat'}
                            >
                                {s.title || 'Đoạn Chat'}
                            </Link>
                            <button
                                onClick={(e) => { e.preventDefault(); e.stopPropagation(); setMenuOpenId(menuOpenId === s.id ? null : s.id) }}
                                className="absolute right-1 top-1/2 -translate-y-1/2 hidden group-hover:inline-flex h-7 w-7 items-center justify-center rounded hover:bg-gray-200 dark:hover:bg-gray-700"
                                title="Tùy chọn"
                            >
                                <MoreVertical className="h-4 w-4" />
                            </button>
                            {menuOpenId === s.id && (
                                <div className="absolute right-2 top-7 z-10 w-40 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg py-1">
                                    <button
                                        onClick={(e) => { e.preventDefault(); e.stopPropagation(); openConfirm(s.id) }}
                                        className="w-full text-left px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-gray-800 flex items-center gap-2"
                                    >
                                        <Trash2 className="h-4 w-4" /> Xóa đoạn chat
                                    </button>
                                </div>
                            )}
                        </li>
                    ))}
                </ul>
            )}

            {/* Confirm Delete Modal */}
            {confirmOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center">
                    <div className="absolute inset-0 bg-black/30" onClick={closeConfirm} />
                    <div className="relative bg-white dark:bg-zinc-900 rounded-lg shadow-xl w-full max-w-md mx-4 p-6">
                        <button className="absolute right-3 top-3 text-gray-400 hover:text-gray-600" onClick={closeConfirm}>
                            <X className="h-4 w-4" />
                        </button>
                        <div className="flex items-center gap-3">
                            <div className="h-10 w-10 rounded-full bg-red-100 flex items-center justify-center text-red-600">
                                <AlertTriangle className="h-5 w-5" />
                            </div>
                            <div>
                                <h3 className="text-lg font-semibold">Xóa phiên chat?</h3>
                                <p className="text-sm text-gray-600">Hành động này sẽ xóa toàn bộ tin nhắn trong phiên và không thể hoàn tác.</p>
                            </div>
                        </div>
                        <div className="mt-5 flex justify-end gap-3">
                            <button onClick={closeConfirm} disabled={deleting} className="px-4 py-2 rounded border bg-white text-gray-700 hover:bg-gray-50 disabled:opacity-50">Hủy</button>
                            <button onClick={() => pendingId && handleDelete(pendingId)} disabled={deleting} className="px-4 py-2 rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50">
                                {deleting ? 'Đang xóa...' : 'Xóa'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}


interface LayoutProps {
    children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
    const { user, logout } = useAuth()
    const { brandingConfig } = useBranding()
    const navigate = useNavigate()

    // Theme (light/dark)
    const [theme, setTheme] = React.useState<'light' | 'dark'>(() => {
        const stored = localStorage.getItem('theme') as 'light' | 'dark' | null
        if (stored) return stored
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        return prefersDark ? 'dark' : 'light'
    })

    React.useEffect(() => {
        const root = document.documentElement
        if (theme === 'dark') {
            root.classList.add('dark')
        } else {
            root.classList.remove('dark')
        }
        localStorage.setItem('theme', theme)
    }, [theme])

    const toggleTheme = () => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))

    // Mobile sidebar state
    const [isSidebarOpen, setSidebarOpen] = React.useState(false)

    const handleLogout = () => {
        logout()
        navigate('/login')
    }

    const userNavItems = [
        { name: 'Đoạn Chat Mới', href: '/dashboard', icon: MessageSquare },
        { name: 'Lịch sử chat', href: '/dashboard/history', icon: FileText },
        { name: 'Hồ sơ', href: '/dashboard/profile', icon: User },
    ]

    const adminNavItems = [
        { name: 'Dashboard', href: '/admin', icon: BarChart3 },
        { name: 'Quản lý User', href: '/admin/users', icon: Users },
        { name: 'Quản lý dữ liệu', href: '/admin/documents', icon: FileText },
    ]

    const navItems = user?.is_admin ? adminNavItems : userNavItems

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-[#0a0a0a]">
            {/* Header */}
            <header className="sticky top-0 z-50 bg-white/90 backdrop-blur supports-[backdrop-filter]:bg-white/70 shadow-sm border-b dark:bg-[#0d0d0d]/90 dark:supports-[backdrop-filter]:bg-[#0d0d0d]/70 dark:border-gray-800">
                <div className="max-w-7xl mx-auto px-3 sm:px-4 md:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center gap-2">
                            {/* Mobile menu button */}
                            <button
                                className="md:hidden inline-flex items-center justify-center h-9 w-9 rounded-md ring-1 ring-gray-200 bg-white hover:bg-gray-50 text-gray-700"
                                onClick={() => setSidebarOpen(true)}
                                aria-label="Mở menu"
                            >
                                <Menu className="h-5 w-5" />
                            </button>

                            <Link to="/dashboard" className="flex items-center space-x-2">
                                {brandingConfig?.brand_logo_url ? (
                                    <img
                                        src={brandingConfig.brand_logo_url}
                                        alt={brandingConfig.brand_name || 'Logo'}
                                        style={{ height: brandingConfig.brand_logo_height || '32px' }}
                                    />
                                ) : null}
                                <span className="text-lg sm:text-xl font-extrabold" style={{ color: 'var(--brand-title-color, #308748)' }}>
                                    {brandingConfig?.brand_name || 'Dalat Hasfarm AI Chat'}
                                </span>
                                <span className="hidden sm:inline-block h-2.5 w-2.5 rounded-full bg-secondary-500"></span>
                            </Link>
                        </div>

                        <div className="flex items-center space-x-3 sm:space-x-4">
                            <ThemeSwitch theme={theme} onToggle={toggleTheme} sizePx={12} />
                            <span className="hidden sm:inline text-sm text-gray-700 dark:text-gray-200">
                                Xin chào, {user?.full_name}
                            </span>
                            {user?.is_admin && (
                                <span className="hidden sm:inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-secondary-100 text-secondary-700">
                                    Admin
                                </span>
                            )}
                            <button
                                onClick={handleLogout}
                                className="flex items-center space-x-1 text-gray-600 hover:text-gray-900 dark:text-gray-200 dark:hover:text-white"
                            >
                                <LogOut className="h-4 w-4" />
                                <span className="hidden sm:inline">Đăng xuất</span>
                            </button>
                        </div>
                    </div>
                </div>
                {/* Accent bar under header */}
                <div className="brand-accent-bar"></div>
                <div className="h-px bg-gray-200 dark:bg-gray-800"></div>
            </header>

            {/* Mobile Drawer */}
            {isSidebarOpen && (
                <div className="fixed inset-0 z-50 md:hidden">
                    <div className="absolute inset-0 bg-black/30" onClick={() => setSidebarOpen(false)} />
                    <nav className="absolute left-0 top-0 h-full w-72 bg-white dark:bg-[#0d0d0d] shadow-lg p-4 overflow-y-auto">
                        <div className="mb-4 flex items-center justify-between">
                            <span className="text-sm font-semibold text-gray-600 dark:text-gray-300">Menu</span>
                            <button onClick={() => setSidebarOpen(false)} className="h-8 w-8 inline-flex items-center justify-center rounded hover:bg-gray-100 dark:hover:bg-gray-800">
                                <X className="h-4 w-4" />
                            </button>
                        </div>
                        <div className="space-y-2">
                            {navItems.map((item) => {
                                const Icon = item.icon
                                return (
                                    <Link key={item.name} to={item.href} onClick={() => setSidebarOpen(false)}
                                        className="flex items-center gap-3 px-3 py-2 rounded-lg text-gray-700 hover:bg-primary-50 hover:text-primary-700 dark:text-gray-200 dark:hover:bg-gray-800">
                                        <Icon className="h-5 w-5" />
                                        <span>{item.name}</span>
                                    </Link>
                                )
                            })}
                            {user && !user.is_admin && (
                                <div className="mt-6">
                                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-3 mb-2">Đoạn Chat</h4>
                                    <ChatSessionsList />
                                </div>
                            )}
                        </div>
                    </nav>
                </div>
            )}

            <div className="flex">
                {/* Sidebar (desktop) */}
                <nav className="hidden md:block fixed top-[4.6rem] left-0 w-64 bg-white shadow-sm dark:bg-[#0d0d0d] dark:border-r dark:border-gray-800 h-[calc(100vh-4.6rem)] overflow-y-auto z-40">
                    <div className="p-4">
                        <nav className="space-y-2">
                            {navItems.map((item) => {
                                const Icon = item.icon
                                return (
                                    <Link
                                        key={item.name}
                                        to={item.href}
                                        className="flex items-center space-x-3 px-3 py-2 text-gray-700 rounded-lg hover:bg-primary-50 hover:text-primary-700 transition-colors dark:text-gray-200 dark:hover:bg-gray-800 dark:hover:text-white"
                                    >
                                        <Icon className="h-5 w-5" />
                                        <span>{item.name}</span>
                                    </Link>
                                )
                            })}

                            {/* Chat sessions list inside main sidebar */}
                            {user && !user.is_admin && (
                                <div className="mt-6">
                                    <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-3 mb-2">Đoạn Chat</h4>
                                    <ChatSessionsList />
                                </div>
                            )}
                        </nav>
                    </div>
                </nav>

                {/* Main content */}
                <main className="flex-1 p-4 sm:p-6 text-gray-900 dark:bg-[#0a0a0a] dark:text-gray-100 h-[calc(100vh-4.6rem)] md:ml-64 overflow-y-auto">
                    {children}
                </main>
            </div>
        </div>
    )
}

export default Layout