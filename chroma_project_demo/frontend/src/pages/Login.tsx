import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Eye, EyeOff, User, Lock } from 'lucide-react'
import toast from 'react-hot-toast'

import Aurora from '../components/Aurora'

const Login: React.FC = () => {
    const [username, setUsername] = useState('')
    const [password, setPassword] = useState('')
    const [showPassword, setShowPassword] = useState(false)
    const [loading, setLoading] = useState(false)

    const { login } = useAuth()
    const navigate = useNavigate()

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)

        try {
            await login(username, password)
            toast.success('Đăng nhập thành công!')
            navigate('/dashboard')
        } catch (error: any) {
            toast.error(error.message || 'Đăng nhập thất bại')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="h-dvh relative bg-black overflow-hidden flex items-center justify-center py-8 sm:py-12 px-4 sm:px-6 lg:px-8">
            <div className="absolute inset-0 pointer-events-none">
                <Aurora
                    colorStops={["#48F4AA", "#FCFCFC", "#F4B152"]}
                    blend={0.5}
                    amplitude={1.0}
                    speed={0.5}
                />
            </div>
            <div className="relative z-10 max-w-md w-full space-y-8">
                {/* Logo and Brand */}
                <div className="text-center lg:text-left">
                    <div className="mx-auto h-16 w-16 bg-[#80DB97] rounded-2xl flex items-center justify-center mb-4 shadow-[0_20px_80px_-20px_rgba(128,219,151,0.7)] ring-2 ring-white/10">
                        <div className="w-8 h-8 bg-white rounded-lg flex items-center justify-center shadow-md">
                            <div className="w-4 h-4 bg-[#80DB97] rounded"></div>
                        </div>
                    </div>
                    <h1 className="text-2xl font-bold text-[#80DB97] mb-1">
                        Dalat Hasfarm AI
                    </h1>
                    <p className="text-gray-500 text-sm">
                        Trợ lý AI thông minh cho mọi câu hỏi
                    </p>
                </div>

                {/* Login Card */}
                <div className="relative rounded-3xl p-[1px] bg-gradient-to-br from-emerald-200/60 via-white to-amber-200/60 shadow-[0_30px_120px_-30px_rgba(0,0,0,0.6)]">
                    <div className="bg-white/85 dark:bg-[#0b0b0b]/80 backdrop-blur-xl rounded-3xl border border-white/30 p-8">
                        <div className="text-center mb-6">
                            <h2 className="text-2xl font-extrabold text-gray-900 dark:text-gray-100 mb-1 tracking-tight">Đăng nhập</h2>
                            <p className="text-gray-500 text-sm">Chào mừng bạn trở lại! 👋</p>
                        </div>

                        <form onSubmit={handleSubmit} className="space-y-5">
                            {/* Username Field */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Tên đăng nhập</label>
                                <div className="relative">
                                    <User className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5" />
                                    <input
                                        type="text"
                                        value={username}
                                        onChange={(e) => setUsername(e.target.value)}
                                        className="w-full pl-10 pr-4 py-3 rounded-xl border ring-1 ring-gray-200/70 bg-emerald-50/50 focus:ring-2 focus:ring-emerald-400/60 focus:border-emerald-400 placeholder-gray-400 transition-all duration-200"
                                        placeholder="Nhập tên đăng nhập của bạn"
                                        required
                                    />
                                </div>
                            </div>

                            {/* Password Field */}
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Mật khẩu</label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5" />
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="w-full pl-10 pr-12 py-3 rounded-xl border ring-1 ring-gray-200/70 bg-emerald-50/50 focus:ring-2 focus:ring-emerald-400/60 focus:border-emerald-400 placeholder-gray-400 transition-all duration-200"
                                        placeholder="Nhập mật khẩu của bạn"
                                        required
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                                    >
                                        {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                                    </button>
                                </div>
                            </div>

                            {/* Login Button */}
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full py-3 px-4 rounded-xl font-semibold text-white bg-gradient-to-r from-emerald-400 to-emerald-500 hover:from-emerald-500 hover:to-emerald-600 shadow-lg shadow-emerald-500/30 focus:outline-none focus:ring-2 focus:ring-emerald-400/50 focus:ring-offset-2 transition-all disabled:opacity-50"
                            >
                                {loading ? (
                                    <div className="flex items-center justify-center">
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2"></div>
                                        Đang đăng nhập...
                                    </div>
                                ) : (
                                    'Đăng nhập'
                                )}
                            </button>
                        </form>

                        {/* Links */}
                        <div className="text-center mt-6">
                            <Link to="/forgot-password" className="text-sm text-gray-600 hover:text-emerald-600 transition-colors">Quên mật khẩu?</Link>
                        </div>
                        <div className="text-center mt-3">
                            <Link to="/register" className="text-sm text-gray-600 hover:text-emerald-600 transition-colors">Tạo tài khoản mới</Link>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Login