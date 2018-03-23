# 0. Calculate max id.
id = `ls downloads`.split("\n").map(&:to_i).max.to_i + 1

# 1. Move files.
`find downloads/*/ -type f`.split("\n").each do |img|
  `mv "#{img}" downloads/#{id}`
  puts [id, img.split('/')[1].gsub(' ', '_')].join(',')
  id += 1
end

# Remove directories.
`find downloads/*/ -type d`.split("\n").each do |dir|
  `rm -r "#{dir}"`
end
