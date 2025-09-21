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
        <div className="h-dvh relative bg-black overflow-hidden flex flex-col items-center justify-center py-8 sm:py-12 px-4 sm:px-6 lg:px-8">
            <div className="absolute inset-0 pointer-events-none">
                <Aurora
                    colorStops={["#48F4AA", "#FCFCFC", "#F4B152"]}
                    blend={0.5}
                    amplitude={1.0}
                    speed={0.5}
                />
            </div>
            <div className="relative z-10 w-full max-w-md">
                {/* Brand */}
                <div className="text-center mb-8 animate-fade-in" style={{ animationDelay: '0.1s' }}>
                    <h1 className="text-4xl font-bold text-green-400 mb-2">
                        Dalat Hasfarm AI
                    </h1>
                    <p className="text-gray-300">
                        Trợ lý AI thông minh cho mọi câu hỏi
                    </p>
                </div>

                {/* Login Card */}
                <div className="bg-[#1A1A1A] rounded-3xl shadow-2xl p-8 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
                    <div className="text-center mb-8">
                        <h2 className="text-3xl font-bold text-white mb-1">Đăng nhập</h2>
                        <p className="text-gray-400">Chào mừng bạn trở lại! 👋</p>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-6">
                        {/* Username Field */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">Mã nhân viên (Staff code)</label>
                            <div className="relative">
                                <User className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 w-5 h-5" />
                                <input
                                    type="text"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    className="w-full pl-12 pr-4 py-3 rounded-xl bg-gray-100 border-transparent text-gray-800 placeholder-gray-500 focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
                                    placeholder="Mã nhân viên"
                                    autoComplete="off"
                                    required onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng nhập mã nhân viên')} onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                                />
                            </div>
                        </div>

                        {/* Password Field */}
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">Mật khẩu</label>
                            <div className="relative">
                                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500 w-5 h-5" />
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className="w-full pl-12 pr-12 py-3 rounded-xl bg-gray-100 border-transparent text-gray-800 placeholder-gray-500 focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
                                    placeholder="Mật khẩu"
                                    autoComplete="new-password"
                                    required
                                    onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng nhập mật khẩu')}
                                    onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-4 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-800 transition-colors"
                                >
                                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                                </button>
                            </div>
                        </div>

                        {/* Login Button */}
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 px-4 rounded-xl font-semibold text-white bg-green-500 hover:bg-green-600 shadow-lg shadow-green-500/30 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:ring-offset-2 focus:ring-offset-gray-900 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {loading ? (
                                <div className="flex items-center justify-center">
                                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2"></div>
                                    Đang xử lý...
                                </div>
                            ) : (
                                'Đăng nhập'
                            )}
                        </button>
                    </form>

                    {/* Links */}
                    <div className="text-center mt-8">
                        <Link to="/forgot-password" className="text-sm text-gray-400 hover:text-green-400 transition-colors">Quên mật khẩu?</Link>
                        <p className="text-sm text-gray-400 mt-3">Chưa có tài khoản? <Link to="/register" className="font-semibold text-green-400 hover:text-green-300 transition-colors">Tạo tài khoản mới</Link></p>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default Login