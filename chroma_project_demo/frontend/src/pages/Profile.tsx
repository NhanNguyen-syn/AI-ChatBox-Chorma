import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../services/api'
import toast from 'react-hot-toast'

interface ProfileData {
  username: string
  email: string
  full_name?: string
  is_admin: boolean
  created_at: string
}

const Profile: React.FC = () => {
  const [profile, setProfile] = useState<ProfileData | null>(null)
  const [email, setEmail] = useState('')
  const [fullName, setFullName] = useState('')
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [saving, setSaving] = useState(false)

  const initials = useMemo(() => {
    const src = fullName || profile?.username || ''
    return src
      .split(' ')
      .filter(Boolean)
      .slice(0, 2)
      .map((s) => s[0]?.toUpperCase())
      .join('') || 'U'
  }, [fullName, profile?.username])

  const load = async () => {
    try {
      const res = await api.get('/users/profile')
      setProfile(res.data)
      setEmail(res.data.email || '')
      setFullName(res.data.full_name || '')
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Không thể tải hồ sơ')
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    setSaving(true)
    try {
      const payload: any = { email, full_name: fullName }
      if (newPassword) {
        payload.current_password = currentPassword
        payload.new_password = newPassword
      }
      await api.put('/users/profile', payload)
      toast.success('Cập nhật hồ sơ thành công')
      if (newPassword) {
        setCurrentPassword('')
        setNewPassword('')
      }
      await load()
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Cập nhật thất bại')
    } finally {
      setSaving(false)
    }
  }

  if (!profile) {
    return (
      <div className="bg-white dark:bg-zinc-900 rounded-lg p-6 shadow animate-pulse">Đang tải hồ sơ...</div>
    )
  }

  return (
    <div className="w-full space-y-6">
      {/* Header card */}
      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow p-6 flex items-start gap-4">
        <div className="h-14 w-14 shrink-0 rounded-full bg-gradient-to-br from-primary-500 to-primary-700 text-white grid place-content-center text-xl font-semibold">
          {initials}
        </div>
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-xl font-semibold">{profile.full_name || profile.username}</h1>
            <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${profile.is_admin ? 'bg-amber-100 text-amber-800 dark:bg-amber-500/20 dark:text-amber-200' : 'bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-200'}`}>
              {profile.is_admin ? 'Admin' : 'User'}
            </span>
          </div>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            <span className="mr-3">{profile.email}</span>
            <span className="text-gray-400">•</span>
            <span className="ml-3">Tham gia: {new Date(profile.created_at).toLocaleDateString()}</span>
          </div>
        </div>
        <div className="mt-2">
          <button
            onClick={save}
            disabled={saving}
            className="px-4 py-2 rounded-lg bg-primary-600 hover:bg-primary-700 text-white disabled:opacity-50"
          >
            {saving ? 'Đang lưu...' : 'Lưu thay đổi'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
        {/* Account info */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl shadow p-6 h-full">
          <h2 className="text-base font-semibold mb-4">Thông tin tài khoản</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">Username</label>
              <input
                value={profile.username}
                readOnly
                className="w-full px-3 py-2 border rounded-lg bg-gray-50 dark:bg-zinc-800 dark:border-zinc-700"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">Email</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:bg-zinc-800 dark:border-zinc-700"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">Họ và tên</label>
              <input
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:bg-zinc-800 dark:border-zinc-700"
              />
            </div>
          </div>
        </div>

        {/* Change password */}
        <div className="bg-white dark:bg-zinc-900 rounded-xl shadow p-6 h-full">
          <h2 className="text-base font-semibold mb-4">Đổi mật khẩu</h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">Mật khẩu hiện tại</label>
              <input
                type="password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:bg-zinc-800 dark:border-zinc-700"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-300 mb-1">Mật khẩu mới</label>
              <input
                type="password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="w-full px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:bg-zinc-800 dark:border-zinc-700"
              />
            </div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
            Để đổi mật khẩu, vui lòng nhập cả mật khẩu hiện tại và mật khẩu mới.
          </p>
        </div>
      </div>

      {/* Save button footer for small screens */}
      <div className="md:hidden">
        <button onClick={save} disabled={saving} className="w-full px-4 py-2 rounded-lg bg-primary-600 text-white disabled:opacity-50">
          {saving ? 'Đang lưu...' : 'Lưu thay đổi'}
        </button>
      </div>
    </div>
  )
}

export default Profile

