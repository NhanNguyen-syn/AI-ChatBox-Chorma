import React, { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../services/api'
import toast from 'react-hot-toast'

interface AdminUserDetail {
  id: string
  username: string
  email: string | null
  full_name: string | null
  phone: string | null
  role: string | null
  department: string | null
  account_status: 'active' | 'inactive' | 'suspended' | null
  is_admin: boolean
  is_active: boolean
  created_at: string
  last_login: string | null
  chat_count: number
}

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300',
  inactive: 'bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300',
  suspended: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/40 dark:text-yellow-300',
}

const AdminUserEdit: React.FC = () => {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [user, setUser] = useState<AdminUserDetail | null>(null)
  const [showPassword, setShowPassword] = useState(false)
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')

  const [form, setForm] = useState({
    full_name: '',
    email: '',
    phone: '',
    username: '',
    role: 'user',
    department: '',
    account_status: 'active' as 'active' | 'inactive' | 'suspended',
  })

  const isSuperAdmin = (u?: AdminUserDetail | null) => {
    const uname = (u?.username || '').toLowerCase()
    return uname === 'admin' || uname === 'sg0510'
  }

  useEffect(() => {
    const load = async () => {
      try {
        const res = await api.get(`/admin/users/${id}`)
        const u: AdminUserDetail = res.data
        setUser(u)
        setForm({
          full_name: u.full_name || '',
          email: u.email || '',
          phone: u.phone || '',
          username: u.username || '',
          role: (u.role as any) || (u.is_admin ? 'admin' : 'user'),
          department: u.department || '',
          account_status: (u.account_status as any) || (u.is_active ? 'active' : 'inactive'),
        })
      } catch (e: any) {
        toast.error(e?.response?.data?.detail || 'Không tải được thông tin user')
        navigate('/admin/users')
      } finally {
        setLoading(false)
      }
    }
    if (id) load()
  }, [id])

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target
    setForm((f) => ({ ...f, [name]: value }))
  }

  const save = async () => {
    // Password validation
    if (password) {
        if (password !== confirmPassword) {
            toast.error('Mật khẩu xác nhận không khớp!')
            return
        }
    }

    setSaving(true)
    try {
        const payload: any = { ...form };
        if (password) {
            payload.new_password = password;
        }

        await api.put(`/admin/users/${id}`, payload)
        toast.success('Cập nhật người dùng thành công')
        navigate('/admin/users')
    } catch (e: any) {
        toast.error(e?.response?.data?.detail || 'Lưu thất bại')
    } finally {
        setSaving(false)
    }
}

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (!user) return null

  const lockSuper = isSuperAdmin(user)
  const status = (form.account_status || 'active').toLowerCase()

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold">Chỉnh sửa User</h2>
            <p className="text-gray-500">@{user.username}</p>
          </div>
          <div className="space-x-2">
            <button className="btn-secondary" onClick={() => navigate(-1)}>Quay lại</button>
            <button className="btn-primary" onClick={save} disabled={saving}>Lưu thay đổi</button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
          <h3 className="font-medium mb-4">Thông tin cơ bản</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Họ và tên</label>
              <input className="input-field w-full" name="full_name" value={form.full_name} onChange={handleChange} />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Email</label>
              <input className="input-field w-full" name="email" value={form.email} onChange={handleChange} />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Số điện thoại (tùy chọn)</label>
              <input className="input-field w-full" name="phone" value={form.phone} onChange={handleChange} />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Staff code / Mã nhân sự</label>
              <input className="input-field w-full" name="username" value={form.username} onChange={handleChange} disabled={lockSuper} />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
          <h3 className="font-medium mb-4">Quyền và trạng thái</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Vai trò</label>
              <select className="input-field w-full" name="role" value={form.role} onChange={handleChange} disabled={lockSuper}>
                <option value="admin">Admin</option>
                <option value="manager">Manager</option>
                <option value="user">User</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Phòng ban / Bộ phận</label>
              <input className="input-field w-full" name="department" value={form.department} onChange={handleChange} />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="block text-sm text-gray-600">Trạng thái tài khoản</label>
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${STATUS_COLORS[status] || ''}`}>
                  {status === 'active' ? 'Hoạt động' : status === 'inactive' ? 'Bị khóa' : 'Tạm ngưng'}
                </span>
              </div>
              <div className="inline-flex border rounded-md overflow-hidden bg-white dark:bg-transparent dark:border-gray-700">
                <button
                  type="button"
                  disabled={lockSuper}
                  onClick={() => setForm((f) => ({ ...f, account_status: 'active' }))}
                  className={`px-3 py-2 text-sm font-medium border-r focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 ${
                    status === 'active'
                      ? 'bg-green-50 text-green-700 border-green-300'
                      : 'bg-white text-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800'
                  } ${lockSuper ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  Active
                </button>
                <button
                  type="button"
                  disabled={lockSuper}
                  onClick={() => setForm((f) => ({ ...f, account_status: 'inactive' }))}
                  className={`px-3 py-2 text-sm font-medium border-r focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 ${
                    status === 'inactive'
                      ? 'bg-orange-50 text-orange-700 border-orange-300'
                      : 'bg-white text-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800'
                  } ${lockSuper ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  Inactive
                </button>
                <button
                  type="button"
                  disabled={lockSuper}
                  onClick={() => setForm((f) => ({ ...f, account_status: 'suspended' }))}
                  className={`px-3 py-2 text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 ${
                    status === 'suspended'
                      ? 'bg-yellow-50 text-yellow-700 border-yellow-300'
                      : 'bg-white text-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800'
                  } ${lockSuper ? 'opacity-60 cursor-not-allowed' : ''}`}
                >
                  Suspended
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
              <div className="bg-gray-50 dark:bg-gray-800/40 rounded-md p-3">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-green-100 text-green-700 flex items-center justify-center shrink-0">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path d="M12 8a.75.75 0 01.75.75V12c0 .199-.079.39-.22.53l-2 2a.75.75 0 11-1.06-1.06l1.72-1.72V8.75A.75.75 0 0112 8z"/><path fillRule="evenodd" d="M12 2.25c-5.385 0-9.75 4.365-9.75 9.75s4.365 9.75 9.75 9.75 9.75-4.365 9.75-9.75S17.385 2.25 12 2.25zm-8.25 9.75a8.25 8.25 0 1116.5 0 8.25 8.25 0 01-16.5 0z" clipRule="evenodd"/></svg>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs uppercase tracking-wide">Ngày tạo tài khoản</div>
                    <div className="font-medium">{new Date(user.created_at).toLocaleString('vi-VN')}</div>
                  </div>
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-800/40 rounded-md p-3">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-700 flex items-center justify-center shrink-0">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4"><path d="M6.75 3A.75.75 0 016 3.75V5H4.5A2.25 2.25 0 002.25 7.25v11A2.25 2.25 0 004.5 20.5h15a2.25 2.25 0 002.25-2.25v-11A2.25 2.25 0 0019.5 5H18V3.75a.75.75 0 00-1.5 0V5h-9V3.75A.75.75 0 006.75 3z"/><path d="M20.25 8.5H3.75v9a.75.75 0 00.75.75h15a.75.75 0 00.75-.75v-9z"/></svg>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs uppercase tracking-wide">Lần đăng nhập gần nhất</div>
                    <div className="font-medium">{user.last_login ? new Date(user.last_login).toLocaleString('vi-VN') : '—'}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Password Reset Section */}
      <div className="bg-white rounded-lg shadow p-6 dark:bg-[#0f0f0f] dark:border dark:border-gray-800">
        <h3 className="font-medium mb-4">Đặt lại mật khẩu</h3>
        <button type="button" className="btn-secondary" onClick={() => setShowPassword((s) => !s)}>
          {showPassword ? 'Ẩn ô đổi mật khẩu' : 'Đổi mật khẩu'}
        </button>
        {showPassword && (
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <label className="block text-sm text-gray-600 mb-1">Mật khẩu mới</label>
              <input
                type="password"
                className="input-field w-full"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Nhập mật khẩu mới"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 mb-1">Xác nhận mật khẩu mới</label>
              <input
                type="password"
                className="input-field w-full"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Nhập lại mật khẩu mới"
              />
            </div>
          </div>
        )}
      </div>

    </div>
  )
}

export default AdminUserEdit

