import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../services/api'
import toast from 'react-hot-toast'

const ForgotPassword: React.FC = () => {
    const [email, setEmail] = useState('')
    const [loading, setLoading] = useState(false)
    const [submitted, setSubmitted] = useState(false)

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setLoading(true)
        try {
            await api.post('/auth/request-password-reset', { email: email.trim() })
            setSubmitted(true)
            toast.success('Nếu email của bạn tồn tại trong hệ thống, một liên kết đặt lại mật khẩu đã được gửi.')
        } catch (error: any) {
            // Dù có lỗi, vẫn hiển thị thông báo chung để tránh lộ thông tin
            setSubmitted(true)
            toast.success('Nếu email của bạn tồn tại trong hệ thống, một liên kết đặt lại mật khẩu đã được gửi.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
            <div className="max-w-md w-full space-y-8">
                <div>
                    <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
                        Quên mật khẩu
                    </h2>
                    <p className="mt-2 text-center text-sm text-gray-600">
                        <Link to="/login" className="font-medium text-primary-600 hover:text-primary-500">
                            Quay lại đăng nhập
                        </Link>
                    </p>
                </div>

                {submitted ? (
                    <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
                        Nếu email của bạn tồn tại trong hệ thống, một liên kết đặt lại mật khẩu đã được gửi đến.
                    </div>
                ) : (
                    <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
                        <div>
                            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                                Email
                            </label>
                            <input
                                id="email"
                                name="email"
                                type="email"
                                required
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                className="input-field mt-1 w-full"
                                placeholder="Nhập email của bạn"
                            />
                        </div>
                        <button
                            type="submit"
                            disabled={loading}
                            className="btn-primary w-full flex justify-center"
                        >
                            {loading ? 'Đang gửi yêu cầu...' : 'Gửi yêu cầu'}
                        </button>
                    </form>
                )}
            </div>
        </div>
    )
}

export default ForgotPassword

