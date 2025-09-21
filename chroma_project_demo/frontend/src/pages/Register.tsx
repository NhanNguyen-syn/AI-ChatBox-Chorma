import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Shield, Eye, EyeOff } from 'lucide-react'
import toast from 'react-hot-toast'

const Register: React.FC = () => {
    const [formData, setFormData] = useState({
        staff_code: '',
        email: '',
        full_name: '',
        phone: '',
        password: '',
        confirmPassword: ''
    })
    const [showPassword, setShowPassword] = useState(false)
    const [showConfirmPassword, setShowConfirmPassword] = useState(false)
    const [loading, setLoading] = useState(false)

    const { register } = useAuth()
    const navigate = useNavigate()

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        })
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()

        if (formData.password !== formData.confirmPassword) {
            toast.error('Mật khẩu xác nhận không khớp')
            return
        }

        setLoading(true)

        try {
            await register(
                formData.staff_code.trim(),
                formData.email.trim(),
                formData.password,
                formData.full_name.trim(),
                formData.phone.trim()
            )
            toast.success('Đăng ký thành công! Vui lòng đăng nhập.')
            navigate('/login')
        } catch (error: any) {
            toast.error(error.message || 'Đăng ký thất bại')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full space-y-8">
                <div>
                    <div className="mx-auto h-12 w-12 flex items-center justify-center rounded-full bg-primary-100">
                        <Shield className="h-6 w-6 text-primary-600" />
                    </div>
                    <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
                        Đăng ký tài khoản mới
                    </h2>
                    <p className="mt-2 text-center text-sm text-gray-600">
                        Hoặc{' '}
                        <Link to="/login" className="font-medium text-primary-600 hover:text-primary-500">
                            đăng nhập vào tài khoản hiện có
                        </Link>
                    </p>
                </div>

                <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
                    <div className="space-y-4">
                        <div>
                            <label htmlFor="staff_code" className="block text-sm font-medium text-gray-700">
                                Mã nhân viên (Staff code)
                            </label>
                            <input
                                id="staff_code"
                                name="staff_code"
                                type="text"
                                required
                                value={formData.staff_code}
                                onChange={handleChange}
                                className="input-field mt-1"
                                placeholder="Nhập mã nhân viên"
                                onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng nhập mã nhân viên')}
                                onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                            />
                        </div>

                        <div>
                            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                                Email
                            </label>
                            <input
                                id="email"
                                name="email"
                                type="email"
                                required
                                value={formData.email}
                                onChange={handleChange}
                                className="input-field mt-1"
                                placeholder="Nhập email"
                                onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng nhập email')}
                                onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                            />
                        </div>

                        <div>
                            <label htmlFor="full_name" className="block text-sm font-medium text-gray-700">
                                Họ và tên
                            </label>
                            <input
                                id="full_name"
                                name="full_name"
                                type="text"
                                value={formData.full_name}
                                onChange={handleChange}
                                className="input-field mt-1"
                                placeholder="Nhập họ và tên"
                            />
                        </div>

                        <div>
                            <label htmlFor="phone" className="block text-sm font-medium text-gray-700">
                                Số điện thoại
                            </label>
                            <input
                                id="phone"
                                name="phone"
                                type="text"
                                required
                                value={formData.phone}
                                onChange={handleChange}
                                className="input-field mt-1"
                                placeholder="Nhập số điện thoại"
                                onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng nhập số điện thoại')}
                                onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                            />
                        </div>

                        <div>
                            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                                Mật khẩu
                            </label>
                            <div className="relative mt-1">
                                <input
                                    id="password"
                                    name="password"
                                    type={showPassword ? 'text' : 'password'}
                                    required
                                    value={formData.password}
                                    onChange={handleChange}
                                    className="input-field pr-10"
                                    placeholder="Nhập mật khẩu"
                                    onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng nhập mật khẩu')}
                                    onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                                />
                                <button
                                    type="button"
                                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                                    onClick={() => setShowPassword(!showPassword)}
                                >
                                    {showPassword ? (
                                        <EyeOff className="h-5 w-5 text-gray-400" />
                                    ) : (
                                        <Eye className="h-5 w-5 text-gray-400" />
                                    )}
                                </button>
                            </div>
                        </div>

                        <div>
                            <label htmlFor="confirmPassword" className="block text-sm font-medium text-gray-700">
                                Xác nhận mật khẩu
                            </label>
                            <div className="relative mt-1">
                                <input
                                    id="confirmPassword"
                                    name="confirmPassword"
                                    type={showConfirmPassword ? 'text' : 'password'}
                                    required
                                    value={formData.confirmPassword}
                                    onChange={handleChange}
                                    className="input-field pr-10"
                                    placeholder="Nhập lại mật khẩu"
                                    onInvalid={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('Vui lòng xác nhận mật khẩu')}
                                    onInput={(e) => (e.currentTarget as HTMLInputElement).setCustomValidity('')}
                                />
                                <button
                                    type="button"
                                    className="absolute inset-y-0 right-0 pr-3 flex items-center"
                                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                >
                                    {showConfirmPassword ? (
                                        <EyeOff className="h-5 w-5 text-gray-400" />
                                    ) : (
                                        <Eye className="h-5 w-5 text-gray-400" />
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>

                    <div>
                        <button
                            type="submit"
                            disabled={loading}
                            className="btn-primary w-full flex justify-center"
                        >
                            {loading ? 'Đang đăng ký...' : 'Đăng ký'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    )
}

export default Register